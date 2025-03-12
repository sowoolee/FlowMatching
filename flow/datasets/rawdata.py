import os
import collections
import numpy as np
import pickle as pkl

def read_dataset():
    file_paths = [
        '~/Desktop/gait_fast/trot/data.pkl',
        '~/Desktop/gait_fast/bound/data.pkl',
        '~/Desktop/gait_fast/pace/data.pkl',
        '~/Desktop/gait_fast/pronk/data.pkl',
    ]
    dataset = {}
    keys = ['actions', 'observations', 'rewards', 'terminals', 'timeouts']

    for file_path in file_paths:
        path = os.path.expanduser(file_path)
        with open(path, 'rb') as f:
            loaded_data = pkl.load(f)
        for key in keys:
            if key in dataset:
                dataset[key] = np.concatenate([dataset[key], loaded_data[key]], axis=0)
            else:
                dataset[key] = loaded_data[key]

    generate_pos = True
    include_clock = False

    seperate_gait = True

    if not include_clock:
        dataset['observations'] = np.concatenate([dataset['observations'][:,0:13], dataset['observations'][:,18:]], axis=-1)
        print(dataset['observations'].shape)

    if not generate_pos:
        dataset['observations'] = dataset['observations'][:,2:]

    if not seperate_gait:
        dataset['rewards'][:,0] = -1

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        final_timestep = dataset['timeouts'][i]

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1