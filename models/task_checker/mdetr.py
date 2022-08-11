# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MDETR model and criterion classes.
"""
from typing import Dict, Optional

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn

#import util.dist as dist
#from util import box_ops
#from util.metrics import accuracy
from models.task_checker.util.misc import NestedTesnsor, interpolate

from .backbone import build_backbone
# from .matcher import build_matcher
# from .postprocessors import build_postprocessors
# from .segmentation import DETRsegm, dice_loss, sigmoid_focal_loss
from .transformer import build_transformer


class MDETR(nn.Module):
    """ This is the MDETR module that performs modulated object detection """

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        qa_dataset,
        aux_loss=False,
        contrastive_hdim=64,
        contrastive_loss=False,
        # contrastive_align_loss=False,
        split_qa_heads=True,
        # predict_final=False,
    ):
        """Initializes the model.

        Args:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         MDETR can detect in a single image. For COCO, we recommend 100 queries.
            qa_dataset: If not None, train a QA head for the target dataset (THOR_QAD or ?)
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            contrastive_loss: If true, perform image-text contrastive learning
            split_qa_heads: If true, use several head for each question type
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        nb_heads = 4 if qa_dataset == "thor_qad" else 8
        self.qa_embed = nn.Embedding(nb_heads if split_qa_heads else 1, hidden_dim)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.contrastive_loss = contrastive_loss
        if contrastive_loss:
            self.contrastive_projection_image = nn.Linear(hidden_dim, contrastive_hdim, bias=False)
            self.contrastive_projection_text = nn.Linear(
                self.transformer.text_encoder.config.hidden_size, contrastive_hdim, bias=False
            )

        self.qa_dataset = qa_dataset
        self.split_qa_heads = split_qa_heads
        if split_qa_heads:
            # TODO: make this more general
            if qa_dataset == "thor_qad":
                self.answer_type_head = nn.Linear(hidden_dim, 3)
                self.answer_existence_head = nn.Linear(hidden_dim, 1)
                self.answer_preposition_head = nn.Linear(hidden_dim, 37)
                self.answer_material_head = nn.Linear(hidden_dim, 12)
            elif qa_dataset == "alfred":
                self.answer_type_head = nn.Linear(hidden_dim, 6)
                self.answer_existence_head = nn.Linear(hidden_dim, 1)
                # self.answer_pickupable_head = nn.Linear(hidden_dim, 1)
                self.answer_picked_up_head = nn.Linear(hidden_dim, 1)
                self.answer_receptacle_head = nn.Linear(hidden_dim, 1)
                self.answer_opened_head = nn.Linear(hidden_dim, 1)
                self.answer_toggled_on_head = nn.Linear(hidden_dim, 1)
                self.answer_sliced_head = nn.Linear(hidden_dim, 1)
            else:
                assert False, f"Invalid qa dataset {qa_dataset}"
        else:
            if qa_dataset == "thor_qad":
                self.answer_head = nn.Linear(hidden_dim, 51)
            elif qa_dataset == "alfred":
                self.answer_head = nn.Linear(hidden_dim, 1)
            else:
                assert False, f"Invalid qa dataset {qa_dataset}"

    def forward(self, samples: NestedTensor, captions, encode_and_save=True, memory_cache=None):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)

        if encode_and_save:
            assert memory_cache is None
            # print(f'NestedTensor.tensors for backbone: {samples.tensors.shape}')
            features, pos = self.backbone(samples)
            src, mask = features[-1].decompose()

            query_embed = self.query_embed.weight
            query_embed = torch.cat([query_embed, self.qa_embed.weight], 0)

            memory_cache = self.transformer(
                self.input_proj(src),
                mask,
                query_embed,
                pos[-1],
                captions,
                encode_and_save=True,
                text_memory=None,
                img_memory=None,
                text_attention_mask=None,
            )

            if self.contrastive_loss:
                memory_cache["text_pooled_op"] = self.contrastive_projection_text(memory_cache["text_pooled_op"])
                memory_cache["img_pooled_op"] = self.contrastive_projection_image(memory_cache["img_pooled_op"])

            return memory_cache

        else:
            assert memory_cache is not None
            # hs.shape = (num_layers=6, b, num_queries + nb_heads, 256)
            hs = self.transformer(
                mask=memory_cache["mask"],
                query_embed=memory_cache["query_embed"],
                pos_embed=memory_cache["pos_embed"],
                encode_and_save=False,
                text_memory=memory_cache["text_memory_resized"],
                img_memory=memory_cache["img_memory"],
                text_attention_mask=memory_cache["text_attention_mask"],
            )
            out = {}
            if self.split_qa_heads:
                if self.qa_dataset == "thor_qad":
                    answer_embeds = hs[0, :, -4:]
                    hs = hs[:, :, :-4]
                    out["pred_answer_type"] = self.answer_type_head(answer_embeds[:, 0])
                    out["pred_answer_existence"] = self.answer_existence_head(answer_embeds[:, 1]).squeeze(-1)
                    out["pred_answer_preposition"] = self.answer_preposition_head(answer_embeds[:, 2])
                    out["pred_answer_material"] = self.answer_material_head(answer_embeds[:, 3])
                elif self.qa_dataset == "alfred":
                    answer_embeds = hs[0, :, -7:]
                    hs = hs[:, :, :-7]
                    out["pred_answer_type"] = self.answer_type_head(answer_embeds[:, 0])
                    out["pred_answer_existence"] = self.answer_existence_head(answer_embeds[:, 1]).squeeze(-1)
                    # out["pred_answer_pickupable"] = self.answer_pickupable_head(answer_embeds[:, 2]).squeeze(-1)
                    out["pred_answer_picked_up"] = self.answer_picked_up_head(answer_embeds[:, 2]).squeeze(-1)
                    out["pred_answer_receptacle"] = self.answer_receptacle_head(answer_embeds[:, 3]).squeeze(-1)
                    out["pred_answer_opened"] = self.answer_opened_head(answer_embeds[:, 4]).squeeze(-1)
                    out["pred_answer_toggled_on"] = self.answer_toggled_on_head(answer_embeds[:, 5]).squeeze(-1)
                    out["pred_answer_sliced"] = self.answer_sliced_head(answer_embeds[:, 6]).squeeze(-1)
                else:
                    assert False, f"Invalid qa dataset {self.qa_dataset}"
            else:
                answer_embeds = hs[0, :, -1]
                hs = hs[:, :, :-1]
                if self.qa_dataset == "alfred":
                    out["pred_answer"] = self.answer_head(answer_embeds).squeeze(-1)
                else:
                    out["pred_answer"] = self.answer_head(answer_embeds)

            # class_embed.weight.shape = (d_model=256, num_classes+1=255+1)
            # output_class.shape = (num_layers=6, b, num_queries=100, num_classes+1=256)
            outputs_class = self.class_embed(hs)
            # bbox_embed(hs).shape = (num_layers=6, b, num_queries=100, 4)
            outputs_coord = self.bbox_embed(hs).sigmoid()
            out.update(
                {
                    "pred_logits": outputs_class[-1],
                    "pred_boxes": outputs_coord[-1],
                }
            )
            if self.aux_loss:
                # TODO: compare and check
                out["aux_outputs"] = [
                    {
                        "pred_logits": a,
                        "pred_boxes": b,
                    }
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                ]
            return out


class ContrastiveCriterion(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, pooled_text, pooled_image):

        normalized_text_emb = F.normalize(pooled_text, p=2, dim=1)
        normalized_img_emb = F.normalize(pooled_image, p=2, dim=1)

        logits = torch.mm(normalized_img_emb, normalized_text_emb.t()) / self.temperature
        labels = torch.arange(logits.size(0)).to(pooled_image.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        loss = (loss_i + loss_t) / 2.0
        return loss


class QACriterionTHORQAD(nn.Module):
    def __init__(self, split_qa_heads):
        super().__init__()
        self.split_qa_heads = split_qa_heads

    def forward(self, output, answers):
        loss = {}
        if not self.split_qa_heads:
            loss["loss_answer_total"] = F.cross_entropy(output["pred_answer"], answers["answer"], reduction="mean")
            acc_total = (output["pred_answer"].argmax(-1) == answers["answer"])
            loss["accuracy_answer_total"] = acc_total.float().mean()
            return loss

        device = output["pred_answer_type"].device

        loss["loss_answer_type"] = F.cross_entropy(output["pred_answer_type"], answers["answer_type"])
        acc_type = (output["pred_answer_type"].argmax(-1) == answers["answer_type"])
        loss["accuracy_answer_type"] = acc_type.sum() / answers["answer_type"].numel()  # .float().mean()?

        # The order matters, it is consistent with VQADataset.type2id
        is_existence = answers["answer_type"] == 0
        is_preposition = answers["answer_type"] == 1
        is_material = answers["answer_type"] == 2

        # Existence (yes/no) type
        existence_norm = is_existence.sum() if is_existence.any() else 1.0
        loss["loss_answer_existence"] = (
                F.binary_cross_entropy_with_logits(
                    output["pred_answer_existence"], answers["answer_existence"].float(), reduction="none"
                ).masked_fill(~is_existence, 0).sum()
                / existence_norm
        )
        acc_existence = ((output["pred_answer_existence"].sigmoid() > 0.5) == answers["answer_existence"])
        loss["accuracy_answer_existence"] = (
            acc_existence[is_existence].sum()
            / is_existence.sum() if is_existence.any() else torch.as_tensor(1.0, device=device)
        )

        # Preposition type
        preposition_norm = is_preposition.sum() if is_preposition.any() else 1.0
        loss["loss_answer_preposition"] = (
                F.cross_entropy(output["pred_answer_preposition"], answers["answer_preposition"], reduction="none")
                .masked_fill(~is_preposition, 0)
                .sum()
                / preposition_norm
        )
        acc_preposition = (output["pred_answer_preposition"].argmax(-1) == answers["answer_preposition"])
        loss["accuracy_answer_preposition"] = (
            acc_preposition[is_preposition].sum()
            / is_preposition.sum() if is_preposition.any() else torch.as_tensor(1.0, device=device)
        )

        # Material type
        material_norm = is_material.sum() if is_material.any() else 1.0
        loss["loss_answer_material"] = (
                F.cross_entropy(output["pred_answer_material"], answers["answer_material"], reduction="none")
                .masked_fill(~is_material, 0)
                .sum()
                / material_norm
        )
        acc_material = (output["pred_answer_material"].argmax(-1) == answers["answer_material"])
        loss["accuracy_answer_material"] = (
            acc_material[is_material].sum() / is_material.sum()
            if is_material.any() else torch.as_tensor(1.0, device=device)
        )

        loss["accuracy_answer_total"] = (
            acc_type
            * (is_existence * acc_existence + is_preposition * acc_preposition + is_material * acc_material)
        ).sum() / acc_type.numel()

        return loss


class QACriterionAlfred(nn.Module):
    def __init__(self, split_qa_heads):
        super().__init__()
        self.split_qa_heads = split_qa_heads
        self.gl_avg = {}
        self.reset_stats()

    def reset_stats(self):
        """ Should be called before evaluation. """

        if self.split_qa_heads:
            self.gl_avg = {
                "existence": [0, 0],
                # "pickupable": [0, 0],
                "picked_up": [0, 0],
                "receptacle": [0, 0],
                "opened": [0, 0],
                "toggled_on": [0, 0],
                "sliced": [0, 0],
                "ans_type": [0, 0]
            }
        else:
            self.gl_avg = {
                "total": [0, 0]
            }

    def forward(self, output, answers):
        loss = {}
        if not self.split_qa_heads:
            loss["loss_answer_total"] = F.binary_cross_entropy_with_logits(
                output["pred_answer"], answers["answer"].float(), reduction="mean"
            )
            attr_total = (output["pred_answer"].sigmoid() > 0.5) == answers["answer"]
            loss["accuracy_answer_total"] = attr_total.float().mean()

            if not self.training:
                self.gl_avg["total"][0] += attr_total.sum().item()
                self.gl_avg["total"][1] += attr_total.numel()

            return loss

        device = output["pred_answer_type"].device

        loss["loss_answer_type"] = F.cross_entropy(output["pred_answer_type"], answers["answer_type"])
        type_acc = output["pred_answer_type"].argmax(-1) == answers["answer_type"]
        loss["accuracy_answer_type"] = type_acc.float().mean()

        is_existence, acc_existence = \
            QACriterionAlfred._calc_loss_and_acc(device, output, answers, 0, "existence", loss)
        # is_pickupable, acc_pickupable = \
        #     QACriterionAlfred._calc_loss_and_acc(device, output, answers, 1, "pickupable", loss)
        is_picked_up, acc_picked_up = \
            QACriterionAlfred._calc_loss_and_acc(device, output, answers, 1, "picked_up", loss)
        is_receptacle, acc_receptacle = \
            QACriterionAlfred._calc_loss_and_acc(device, output, answers, 2, "receptacle", loss)
        is_opened, acc_opened = \
            QACriterionAlfred._calc_loss_and_acc(device, output, answers, 3, "opened", loss)
        is_toggled_on, acc_toggled_on = \
            QACriterionAlfred._calc_loss_and_acc(device, output, answers, 4, "toggled_on", loss)
        is_sliced, acc_sliced = \
            QACriterionAlfred._calc_loss_and_acc(device, output, answers, 5, "sliced", loss)

        # Since the accuracy_answer_total is not precise (1 / batch_size limits it),
        # let's accumulate results from every batch and calculate real global average
        if not self.training:
            self.gl_avg["ans_type"][0] += type_acc.sum().item()
            self.gl_avg["ans_type"][1] += type_acc.numel()
            self.gl_avg["existence"][0] += acc_existence[is_existence].sum().item()
            self.gl_avg["existence"][1] += is_existence.sum().item()
            # self.gl_avg["pickupable"][0] += acc_pickupable[is_pickupable].sum().item()
            # self.gl_avg["pickupable"][1] += is_pickupable.sum().item()
            self.gl_avg["picked_up"][0] += acc_picked_up[is_picked_up].sum().item()
            self.gl_avg["picked_up"][1] += is_picked_up.sum().item()
            self.gl_avg["receptacle"][0] += acc_receptacle[is_receptacle].sum().item()
            self.gl_avg["receptacle"][1] += is_receptacle.sum().item()
            self.gl_avg["opened"][0] += acc_opened[is_opened].sum().item()
            self.gl_avg["opened"][1] += is_opened.sum().item()
            self.gl_avg["toggled_on"][0] += acc_toggled_on[is_toggled_on].sum().item()
            self.gl_avg["toggled_on"][1] += is_toggled_on.sum().item()
            self.gl_avg["sliced"][0] += acc_sliced[is_sliced].sum().item()
            self.gl_avg["sliced"][1] += is_sliced.sum().item()

        loss["accuracy_answer_total"] = (
            type_acc
            * (is_existence * acc_existence + is_picked_up * acc_picked_up +
               is_receptacle * acc_receptacle + is_opened * acc_opened + is_toggled_on * acc_toggled_on +
               is_sliced * acc_sliced)
        ).sum() / type_acc.numel()

        return loss

    @staticmethod
    def _calc_loss_and_acc(device, output, answers, type_id, type_name, loss):
        is_type = answers["answer_type"] == type_id
        type_norm = is_type.sum() if is_type.any() else torch.as_tensor(1.0, device=device)
        loss[f"loss_answer_{type_name}"] = (
                F.binary_cross_entropy_with_logits(
                    output[f"pred_answer_{type_name}"], answers[f"answer_{type_name}"].float(), reduction="none"
                ).masked_fill(~is_type, 0).sum()
                / type_norm
        )
        acc_type = (output[f"pred_answer_{type_name}"].sigmoid() > 0.5) == answers[f"answer_{type_name}"]
        loss[f"accuracy_answer_{type_name}"] = (
            acc_type[is_type].sum() / type_norm
        )
        return is_type, acc_type


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN) """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(qa_dataset: str, backbone_args: Dict, transformer_args: Dict, mdetr_args: Dict):
    num_classes = 255

    backbone = build_backbone(backbone_args)
    transformer = build_transformer(transformer_args)
    model = MDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=mdetr_args["num_queries"],
        aux_loss=mdetr_args["aux_loss"],
        contrastive_hdim=mdetr_args["contrastive_loss_hdim"],
        contrastive_loss=mdetr_args["contrastive_loss"],
        qa_dataset=qa_dataset,
        split_qa_heads=mdetr_args["split_qa_heads"],
    )

    return model