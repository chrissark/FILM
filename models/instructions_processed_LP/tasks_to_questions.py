import models.instructions_processed_LP.ALFRED_task_helper as alf
import pickle
import json
import string
import argparse
from tqdm import tqdm

prepositions = {
    'armchair': 'on',
    'bed': 'on',
    'bowl': 'in',
    'box': 'in',
    'bathtubbasin': 'in',
    'cabinet': 'in',
    'coffeemachine': 'in',
    'coffeetable': 'on',
    'countertop': 'on',
    'desk': 'on',
    'diningtable': 'on',
    'drawer': 'in',
    'dresser': 'in',
    'fridge': 'in',
    'garbagecan': 'in',
    'handtowelholder': 'on',
    'laundryhamper': 'in',
    'microwave': 'in',
    'mug': 'in',
    'cup': 'in',
    'ottoman': 'on',
    'pan': 'on',
    'plate': 'on',
    'pot': 'in',
    'paintinghanger': 'on',
    'safe': 'in',
    'shelf': 'on',
    'sidetable': 'on',
    'sinkbasin': 'in',
    'sofa': 'on',
    'stoveburner': 'on',
    'tvstand': 'on',
    'toaster': 'in',
    'toilet': 'on',
    'toiletpaperhanger': 'on',
    'towelholder': 'on',
    'cart': 'in'
}


def generate_list_of_actions(pickled_instructions, path_to_json):
    with open(pickled_instructions, 'rb') as f:
        test_dict = pickle.load(f)
    with open(path_to_json, 'r') as f:
        traj_data = json.load(f)
    list_of_actions, categories_in_inst, second_object, caution_pointers = alf.get_list_of_highlevel_actions(
        traj_data, True, test_dict
    )
    print(f'list of actions: {list_of_actions}')

    return list_of_actions


def generate_questions_from_task(task, prev_obj=None):
    obj, action = task[0], task[1]
    questions = []

    if action == 'PickupObject':
        questions.append(f'Is the {obj} close enough to be picked up?')
        questions.append(f'Is the {obj} being held?')

    elif action == 'PutObject':
        if prev_obj:
            questions.append(f'Is the {prev_obj} {prepositions[obj]} the {obj}?')

    elif action in ('OpenObject', 'CloseObject'):
        questions.append(f'Is the {obj} opened?')

    elif action in ('ToggleObjectOn', 'ToggleObjectOff'):
        questions.append(f'Is the {obj} on?')

    elif action == 'SliceObject':
        questions.append(f'Is the {obj} sliced?')

    return questions


def generate_existence_question(obj, output=None):
    if output is None:
        output = []
    return output + [f'Is the {obj} visible?', ]


def generate_questions_from_list_of_actions(list_of_actions):
    # vowels = {'a', 'e', 'i', 'o', 'u', 'y'}
    output = []
    prev_obj = picked_obj = None
    for i, (obj, action) in enumerate(list_of_actions):
        obj = obj.lower()
        if 'sliced' in obj:
            obj = 'sliced ' + obj.replace('sliced', '')
        if obj != prev_obj:
            output.append(generate_existence_question(obj)[0])
        if action == 'PickupObject':
            picked_obj = obj
        if action == 'PutObject':
            output.append(generate_questions_from_task((obj, action), picked_obj)[0])
        else:
            questions = generate_questions_from_task((obj, action), None)
            for ques in questions:
                output.append(ques)
        prev_obj = obj

    return output


def write_questions_to_file(root, path_to_splits, path_to_pickled_data, output_path, split):
    data = json.load(open(path_to_splits + '/oct21.json'))
    output = []
    split_data = data[split]
    for e in tqdm(split_data):
        r_idx = e['repeat_idx']
        task = e['task']
        path_to_json = root + f'/{task}/pp/ann_{r_idx}.json'
        list_of_actions = generate_list_of_actions(path_to_pickled_data, path_to_json)
        output.append({'task_id': task, 'repeat_idx': r_idx, 'tasks': list_of_actions,
                       'questions': generate_questions_from_list_of_actions(list_of_actions)})
    with open(output_path + f'/{split}_questions.json', 'w') as f:
        json.dump(output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Questions from tasks generation.')
    parser.add_argument('--path_to_pickled_data', type=str)
    parser.add_argument('--path_to_splits', type=str, default='splits')
    parser.add_argument('--split', type=str, choices=['valid_seen', 'valid_unseen', 'tests_seen', 'tests_unseen'])
    parser.add_argument('--output_path', type=str, default='.')
    parser.add_argument('--root', type=str, default='json_2.1.0')
    args = parser.parse_args()
    write_questions_to_file(args.root, args.path_to_splits, args.path_to_pickled_data, args.output_path, args.split)
