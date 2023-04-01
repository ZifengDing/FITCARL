import pickle
import os
import argparse
from model.environment import Env
from dataset.baseDataset import baseFSLDataset
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocess', usage='preprocess_data.py [<args>] [-h | --help]')
    parser.add_argument('--data_dir', default='data/ICEWS14/processed_data', type=str, help='Path to data.')
    parser.add_argument('--outfile', default='state_actions_space.pkl', type=str,
                        help='file to save the preprocessed data.')
    parser.add_argument('--store_actions_num', default=0, type=int,
                        help='maximum number of stored neighbors, 0 means store all.')
    args = parser.parse_args()

    trainF = os.path.join(args.data_dir, 'back_metatrain.txt')
    testF = os.path.join(args.data_dir, 'back_metatest.txt')
    statF = os.path.join(args.data_dir, 'stat.txt')
    validF = os.path.join(args.data_dir, 'back_metavalid.txt')
    backF = os.path.join(args.data_dir, 'background_graph_for_rl.txt')
    if not os.path.exists(validF):
        validF = None
    dataset_train = baseFSLDataset(trainF, statF, ismeta=False)
    dataset_valid = baseFSLDataset(validF, statF, ismeta=False)
    dataset_test = baseFSLDataset(testF, statF, ismeta=False)
    dataset_back = baseFSLDataset(backF, statF, ismeta=False)
    config = {
        'num_rel': dataset_train.num_r,
        'num_ent': dataset_train.num_e,
    }
    # create graph by combining meta sets with background
    env_train = Env(dataset_train.allQuadruples, config)
    env_valid = Env(dataset_valid.allQuadruples, config)
    env_test = Env(dataset_test.allQuadruples, config)
    env_back = Env(dataset_back.allQuadruples, config)

    state_actions_space_train = {}
    state_actions_space_valid = {}
    state_actions_space_test = {}
    state_actions_space_back = {}
    timestamps_train = list(dataset_train.get_all_timestamps())
    timestamps_valid = list(dataset_valid.get_all_timestamps())
    timestamps_test = list(dataset_test.get_all_timestamps())
    timestamps_back = list(dataset_back.get_all_timestamps())

    print(args)
    with tqdm(total=len(dataset_back.allQuadruples)) as bar:
        for (head, rel, tail, t) in dataset_back.allQuadruples:
            if (head, t, True) not in state_actions_space_back.keys():
                state_actions_space_back[(head, t, True)] = env_back.get_state_actions_space_complete(head, t, True, args.store_actions_num)
                # state_actions_space_back[(head, t, False)] = env_back.get_state_actions_space_complete(head, t, False, args.store_actions_num)
            if (tail, t, True) not in state_actions_space_back.keys():
                state_actions_space_back[(tail, t, True)] = env_back.get_state_actions_space_complete(tail, t, True, args.store_actions_num)
                # state_actions_space_back[(tail, t, False)] = env_back.get_state_actions_space_complete(tail, t, False, args.store_actions_num)
            bar.update(1)
    save_path = os.path.join(args.data_dir, 'state_actions_space_back.pkl')
    pickle.dump(state_actions_space_back, open(save_path, 'wb'))