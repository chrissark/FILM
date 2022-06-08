import argparse
import pickle


def get_args():
    parser = argparse.ArgumentParser(description='Analyzer of runs stats')

    parser.add_argument("--recs_filename", type=str,
                        help="Specify the desired recordings to analyze")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    result = pickle.load(open(args.recs_filename, 'rb'))
    print(result)
    # num of episodes
    print(f"Number of episodes: {len(result)}")
    # num of succeeded episodes
    successes = sum(s['success'] for s in result)
    print(f"Number of success episodes: {successes}")
    print(f"Accuracy: {successes / len(result): .4f}")
