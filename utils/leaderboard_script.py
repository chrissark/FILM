import pickle
import argparse
import json
from glob import glob
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dn_startswith', type=str, required=True)
#parser.add_argument('--seen', action='store_true')
parser.add_argument('--json_name', type=str)

args = parser.parse_args()

if args.json_name is None:
	args.json_name = args.dn

# returns 'seen' or 'unseen' logs
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

	return total_logs

total_logs_seen = return_logs('seen')
total_logs_unseen = return_logs('unseen')


results = {'test_unseen': total_logs_unseen, 'test_seen': total_logs_seen}
print(f'Num seen episodes: {len(total_logs_seen)}')
print(f'Num unseen episodes: {len(total_logs_unseen)}')     

if not os.path.exists('leaderboard_jsons'):
	os.makedirs('leaderboard_jsons')

save_path = 'leaderboard_jsons/tests_actseqs_' + args.json_name + '.json'
with open(save_path, 'w') as r:
	json.dump(results, r, indent=4, sort_keys=True)


