import pickle
import argparse
import json
from glob import glob
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dn_startswith', type=str, required=True)
parser.add_argument('--scene_names_path', type=str, required=True)
#parser.add_argument('--seen', action='store_true')
parser.add_argument('--json_name', type=str)
parser.add_argument('--scene_descs_path', type=str, required=True)

args = parser.parse_args()


if args.json_name is None:
	args.json_name = args.dn

# returns 'seen' or 'unseen' logs, all scene names
def return_logs(split):
	pickle_globs = glob("results/leaderboard/actseqs_test_" + split + "_" + args.dn_startswith + "*")
	pickles = []
	for g in pickle_globs:
		pickles += pickle.load(open(g, 'rb'))

	total_logs =[]
	for i, t in enumerate(pickles):
		key = list(t.keys())[0]
		actions = t[key]
		trial = key[1]
		total_logs.append({trial:actions})

	for i, t in enumerate(total_logs):
		key = list(t.keys())[0]
		actions = t[key]
		new_actions = []
		for action in actions:
			if action['action'] == 'LookDown_0' or action['action'] == 'LookUp_0':
				pass
			else:
				new_actions.append(action)
		total_logs[i] = {key: new_actions}

   #scene names (tasks id)
	scene_names_json = json.load(open(args.scene_names_path + '/oct21.json', 'r'))
	scene_names = []
	split_data = scene_names_json['tests_' + split]
	for e in split_data:
		r_idx = e['repeat_idx']
		task = e['task']
		path_to_json = args.scene_descs_path + f'/{task}/pp/ann_{r_idx}.json'
		traj_data = json.load(open(path_to_json, 'r'))
		task_id = traj_data['task_id']
		scene_names.append(task_id)

	print(f'Num scene names: {len(scene_names)}')
    

	return total_logs, scene_names

total_logs_seen, scene_names_seen = return_logs('seen')
total_logs_unseen, scene_names_unseen = return_logs('unseen')

#placeholders
if len(total_logs_seen) < 1533:
	print('Number of test_seen episodes is less than 1533. Adding placeholders')
	for i in range(len(total_logs_seen), 1533):   
		total_logs_seen.append({scene_names_seen[i]: [{'action': 'LookDown_15', 'forceAction': True}]})

if len(total_logs_unseen) < 1529:
	print('Number of test_unseen episodes is less than 1529. Adding placeholders')
	for i in range(len(total_logs_unseen), 1529):  
		total_logs_unseen.append({scene_names_unseen[i]: [{'action': 'LookDown_15', 'forceAction': True}]})


results = {'test_unseen': total_logs_unseen, 'test_seen': total_logs_seen}
print(f'Num seen episodes: {len(total_logs_seen)}')
print(f'Num unseen episodes: {len(total_logs_unseen)}')     

if not os.path.exists('leaderboard_jsons'):
	os.makedirs('leaderboard_jsons')

save_path = 'leaderboard_jsons/tests_actseqs_' + args.json_name + '.json'
with open(save_path, 'w') as r:
	json.dump(results, r, indent=4, sort_keys=True)


