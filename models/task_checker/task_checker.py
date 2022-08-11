import os

import torch
import torchvision.transforms as T

from PIL import Image
# import cv2
import numpy as np
from typing import Tuple

from models.task_checker.mdetr import build
from models.instructions_processed_LP.tasks_to_questions \
    import generate_questions_from_task, generate_existence_question


class TaskChecker:
    backbone_args = {
        "hidden_dim": 256,
        "position_embedding": "sine",
        "lr_backbone": 1.4e-6,
        "backbone": "resnet101",
        "dilation": False
    }
    transformer_args = {
        "hidden_dim": 256,
        "dropout": 0.1,
        "nheads": 8,
        "dim_feedforward": 2048,
        "enc_layers": 6,
        "dec_layers": 6,
        "pre_norm": False,
        "pass_pos_and_query": True,
        "text_encoder_type": "roberta-base",
        "freeze_text_encoder": False,
        "contrastive_loss": False,
        "text_encoder_lr": 7e-6
    }
    mdetr_args = {
        "num_queries": 100,
        "aux_loss": False,  # Not equal to default in MDETR
        "contrastive_loss_hdim": 64,
        "contrastive_loss": False,
        "split_qa_heads": True,  # Not equal to default in MDETR
        "optimizer": "adamw",
        "lr": 7e-6,
        "lr_backbone": backbone_args["lr_backbone"],
        "text_encoder_lr": transformer_args["text_encoder_lr"],
        "lr_answer_heads": 1e-4,
        "lr_drop": 35,
        "weight_decay": 1e-4,
        "clip_max_norm": 0.1,
        "fraction_warmup_steps": 0.01,
        "schedule": "linear_with_warmup"
    }
    ques_id2type = [
        "existence", "picked_up", "receptacle",
        "opened", "toggled_on", "sliced"
    ]

    # Standard PyTorch mean-std input image normalization
    # which was also performed during MDETR's training
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def __init__(
            self,
            mdetr_ckpt_path="models/task_checker/BEST_checkpoint.pth",
            debug=False,
    ):
        self.debug = debug
        self.log_dir_path = None
        if self.debug:
            self.image_cnt = 0
        self.prev_steps_taken = -1  # For the hack

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.mdetr = build(
            "alfred", TaskChecker.backbone_args, TaskChecker.transformer_args,
            TaskChecker.mdetr_args
        ).to(self.device)

        # Load the checkpoint (strictly)
        checkpoint = torch.load(mdetr_ckpt_path, map_location="cpu")
        self.mdetr.load_state_dict(checkpoint["model"], strict=True)
        self.mdetr.eval()  # We use the already trained MDETR
        print(f"MDETR was loaded from {mdetr_ckpt_path}")

        if self.debug:
            n_parameters = sum(
                p.numel() for p in self.mdetr.parameters() if p.requires_grad
            )
            print("Number of params:", n_parameters)
            print(f"The device is {self.device}")

    def set_log_dir_path(self, log_dir_path):
        """
        This function is obligatory to call
        in Sem_Exp_Env_Agent_Thor.load_initial_scene()
        """

        # TODO: add conditional creation
        self.log_dir_path = "task_checker_logs/" + log_dir_path
        pictures_path = self.log_dir_path + "pictures/"
        os.makedirs(pictures_path, exist_ok=True)
        self.image_cnt = 0
        self.prev_steps_taken = -1

    def __call__(self, rgb: np.ndarray, subgoal: Tuple, steps_taken: int,
                 approximate_success: bool = False, prev_obj: str = None):
        # Make the object lowercase
        obj = subgoal[0].lower()
        subgoal = (obj, subgoal[1])

        if prev_obj is not None:
            prev_obj = prev_obj.lower()

        # Apply standard transform
        img = TaskChecker.transform(rgb).unsqueeze(0)  # Added batch dimension

        # We check the ability to pick up an object, the existence of an object
        # and success of basic actions

        # in some cases verdict is the inverted answer
        invert_answer = False

        if approximate_success:
            if subgoal[1] == "PutObject":
                assert prev_obj is not None
                question = generate_questions_from_task(subgoal, prev_obj)[0]
            #elif subgoal[1] == "PickupObject":
            #    question = generate_questions_from_task(subgoal)[1]
            elif subgoal[1] in ("ToggleObjectOn", "OpenObject", "Slice"):
                question = generate_questions_from_task(subgoal)[0]
            elif subgoal[1] in ("ToggleObjectOff", "CloseObject"):
                question = generate_questions_from_task(subgoal)[0]
                invert_answer = True
            else:
                return True
        else:
            #if subgoal[1] == "PickupObject":
            #    question = generate_questions_from_task(subgoal)[0]
            if subgoal[1] in ("OpenObject", "ToggleObjectOn", "SliceObject", "PickupObject"):
                question = generate_existence_question(subgoal[0])[0]
            else:
                return True  # Not implemented

        img = img.to(self.device)
        question = [question, ]  # Added batch dimension
        memory_cache = self.mdetr(img, question, encode_and_save=True)
        outputs = self.mdetr(
            img, question, encode_and_save=False, memory_cache=memory_cache
        )

        # We use multi-head version since its performance is better
        ques_type_pred = outputs["pred_answer_type"].argmax(-1).item()
        ques_type_pred = TaskChecker.ques_id2type[ques_type_pred]
        prob = outputs[f"pred_answer_{ques_type_pred}"].sigmoid().item()
        verdict = prob > 0.5

        if self.debug:
            ques_logs_path = self.log_dir_path + "ques_and_ans.txt"
            self.image_cnt += 1
            img_path = self.log_dir_path + f"pictures/{self.image_cnt}.png"

            with open(ques_logs_path, "a") as f:
                f.write(
                    question[0] + f"\t{verdict}" + f"\tprob={prob:.5f}"
                    + f"\tsteps_taken={steps_taken}\n"
                )
                #if self.prev_steps_taken + 1 == steps_taken and not verdict:
                #    f.write(
                #        "Warning! FILM stuck! Explicitly allowing interaction\n"
                #    )
            Image.fromarray(rgb).save(img_path)

        if invert_answer:
            verdict = not verdict

        # This is the hack. When FILM stuck, explicitly allow to interact
        if self.prev_steps_taken + 1 == steps_taken and not verdict:
            verdict = True
        self.prev_steps_taken = steps_taken

        return verdict


# If you want to test `TaskChecker`, run the code below.
# But before running this, update the `obs_path` to the desired observation.
if __name__ == "__main__":
    task_checker = TaskChecker(mdetr_ckpt_path="BEST_checkpoint.pth")

    subgoal = ("Lettuce", "PickupObject")
    obs_path = "104.png"
    img = np.array(Image.open(obs_path))
    print(img.shape, img.dtype)  # (H, W, 3)
    # Repeat all the transformations to get the same obs as in FILM
    # bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # depth = np.ones(img.shape[:2])  # (H, W)
    # depth = np.expand_dims(depth, 2)
    # sem_seg_pred = np.ones((img.shape[0], img.shape[1], 28))
    # obs = np.concatenate((bgr, depth, sem_seg_pred), axis=2).transpose(2, 0, 1)
    # print(obs.shape, obs.dtype)  # (32, H, W)
    obs = img

    print(f"The verdict is: {task_checker(obs, subgoal)}")
    subgoal = ("Lettuce", "OpenObject")
    print(f"The verdict is: {task_checker(obs, subgoal)}")
    subgoal = ("Lettuce", "ToggleOff")
    print(f"The verdict is: {task_checker(obs, subgoal)}")
