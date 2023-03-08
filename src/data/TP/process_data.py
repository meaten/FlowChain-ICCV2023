import sys
sys.path.append('./src')
import os
import numpy as np
import pandas as pd
import dill
import pickle

from data.TP.environment import Environment, Scene, Node, derivative_of

np.random.seed(123)

desired_max_time = 100
pred_indices = [2, 3]
state_dim = 6
frame_diff = 10
desired_frame_diff = 1
dt = 0.4

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    }
}

def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise

def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        x = node.data.position.x.copy()
        y = node.data.position.y.copy()

        x, y = rotate_pc(np.array([x, y]), alpha)

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        data_dict = {('position', 'x'): x,
                     ('position', 'y'): y,
                     ('velocity', 'x'): vx,
                     ('velocity', 'y'): vy,
                     ('acceleration', 'x'): ax,
                     ('acceleration', 'y'): ay}

        node_data = pd.DataFrame(data_dict, columns=data_columns)

        node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep)

        scene_aug.nodes.append(node)
    return scene_aug


def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug


nl = 0
l = 0

raw_path = './src/data/TP/raw_data'
data_folder_name = './src/data/TP/processed_data/'

maybe_makedirs(data_folder_name)

data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])


# Process ETH-UCY
for desired_source in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
    for data_class in ['train', 'val', 'test']:
        env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
        env.attention_radius = attention_radius

        scenes = []
        data_dict_path = os.path.join(data_folder_name, '_'.join([desired_source, data_class]) + '.pkl')

        for subdir, dirs, files in os.walk(os.path.join(raw_path, desired_source, data_class)):
            for file in files:
                if file.endswith('.txt'):
                    input_data_dict = dict()
                    full_data_path = os.path.join(subdir, file)
                    print('At', full_data_path)

                    data = pd.read_csv(full_data_path, sep='\t', index_col=False, header=None)
                    data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
                    data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
                    data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')

                    data['frame_id'] = data['frame_id'] // 10

                    #data['frame_id'] -= data['frame_id'].min()

                    data['node_type'] = 'PEDESTRIAN'
                    data['node_id'] = data['track_id'].astype(str)

                    data.sort_values('frame_id', inplace=True)
                    
                    if desired_source == "eth" and data_class == "test":
                        data['pos_x'] = data['pos_x'] * 0.6
                        data['pos_y'] = data['pos_y'] * 0.6

                    # if data_class == "train":
                    #     #data_gauss = data.copy(deep=True)
                    #     data['pos_x'] = data['pos_x'] + 2 * np.random.normal(0,1)
                    #     data['pos_y'] = data['pos_y'] + 2 * np.random.normal(0,1)

                        #data = pd.concat([data, data_gauss])

                    #data['pos_x'] = data['pos_x'] - data['pos_x'].mean()
                    #data['pos_y'] = data['pos_y'] - data['pos_y'].mean()

                    max_timesteps = data['frame_id'].max()

                    scene = Scene(timesteps=max_timesteps+1, dt=dt, name=file.rstrip('.txt'), aug_func=augment if data_class == 'train' else None)
                    
                    for node_id in pd.unique(data['node_id']):

                        node_df = data[data['node_id'] == node_id]

                        node_values = node_df[['pos_x', 'pos_y']].values

                        if node_values.shape[0] < 2:
                            continue

                        new_first_idx = node_df['frame_id'].iloc[0]

                        x = node_values[:, 0]
                        y = node_values[:, 1]
                        vx = derivative_of(x, scene.dt)
                        vy = derivative_of(y, scene.dt)
                        ax = derivative_of(vx, scene.dt)
                        ay = derivative_of(vy, scene.dt)

                        data_dict = {('position', 'x'): x,
                                     ('position', 'y'): y,
                                     ('velocity', 'x'): vx,
                                     ('velocity', 'y'): vy,
                                     ('acceleration', 'x'): ax,
                                     ('acceleration', 'y'): ay}

                        node_data = pd.DataFrame(data_dict, columns=data_columns)
                        node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data)
                        node.first_timestep = new_first_idx

                        scene.nodes.append(node)
                    if data_class == 'train':
                        scene.augmented = list()
                        angles = np.arange(0, 360, 15) if data_class == 'train' else [0]
                        for angle in angles:
                            scene.augmented.append(augment_scene(scene, angle))

                    print(scene)
                    scenes.append(scene)
        print(f'Processed {len(scenes):.2f} scene for data class {data_class}')

        env.scenes = scenes

        if len(scenes) > 0:
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)


# Process Stanford Drone. Data obtained from Y-Net github repo
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

for data_class in ["train", "test"]:
    data_path = os.path.join(raw_path, "stanford", f"{data_class}_trajnet.pkl")
    print(f"Processing SDD {data_class}")
    data_out_path = os.path.join(data_folder_name, f"sdd_{data_class}.pkl")
    df = pickle.load(open(data_path, "rb"))
    env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
    env.attention_radius = attention_radius

    scenes = []

    group = df.groupby("sceneId")
    for scene, data in group:
        data['frame'] = pd.to_numeric(data['frame'], downcast='integer')
        data['trackId'] = pd.to_numeric(data['trackId'], downcast='integer')

        data['frame'] = data['frame'] // 12

        #data['frame'] -= data['frame'].min()

        data['node_type'] = 'PEDESTRIAN'
        data['node_id'] = data['trackId'].astype(str)

        # apply data scale as same as PECnet
        data['x'] = data['x']/50
        data['y'] = data['y']/50

        # Mean Position
        #data['x'] = data['x'] - data['x'].mean()
        #data['y'] = data['y'] - data['y'].mean()

        max_timesteps = data['frame'].max()

        if len(data) > 0:

            scene = Scene(timesteps=max_timesteps+1, dt=dt, name=scene, aug_func=augment if data_class == 'train' else None)
            n=0
            for node_id in pd.unique(data['node_id']):

                node_df = data[data['node_id'] == node_id]


                if len(node_df) > 1:
                    assert np.all(np.diff(node_df['frame']) == 1)
                    if not np.all(np.diff(node_df['frame']) == 1):
                        import pdb;pdb.set_trace()
                        

                    node_values = node_df[['x', 'y']].values

                    if node_values.shape[0] < 2:
                        continue

                    new_first_idx = node_df['frame'].iloc[0]

                    x = node_values[:, 0]
                    y = node_values[:, 1]
                    vx = derivative_of(x, scene.dt)
                    vy = derivative_of(y, scene.dt)
                    ax = derivative_of(vx, scene.dt)
                    ay = derivative_of(vy, scene.dt)

                    data_dict = {('position', 'x'): x,
                                 ('position', 'y'): y,
                                 ('velocity', 'x'): vx,
                                 ('velocity', 'y'): vy,
                                 ('acceleration', 'x'): ax,
                                 ('acceleration', 'y'): ay}

                    node_data = pd.DataFrame(data_dict, columns=data_columns)
                    node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data)
                    node.first_timestep = new_first_idx

                    scene.nodes.append(node)
            if data_class == 'train':
                scene.augmented = list()
                angles = np.arange(0, 360, 15) if data_class == 'train' else [0]
                for angle in angles:
                    scene.augmented.append(augment_scene(scene, angle))

            print(scene)
            scenes.append(scene)
    env.scenes = scenes

    if len(scenes) > 0:
        with open(data_out_path, 'wb') as f:
            #pdb.set_trace()
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)


# Process simulated dataset
for desired_source in ['simline', 'simfork', 'simcross']:
    for data_class in ['train', 'val', 'test']:
        env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
        env.attention_radius = attention_radius

        scenes = []
        data_dict_path = os.path.join(data_folder_name, '_'.join([desired_source, data_class]) + '.pkl')

        d = {'train': 1000,
             'val': 100,
             'test': 100}
        N_seqs = d[data_class]
        obs_len = 8
        pred_len = 12
        seq_len = obs_len + pred_len + 1  # add 1 for velocity/acceleration caliculation
        theta = np.pi / 3
        std = 0.5
        
        traj_base = []
        for sgn in [1, -1]:
            traj_flat_x = np.arange(-obs_len-1, 0., 1., dtype=float)
            traj_flat_y = np.zeros_like(traj_flat_x, dtype=float)
            traj_flat = np.concatenate([traj_flat_x[:, None], traj_flat_y[:, None]], axis=1)
            
            if desired_source == 'simline':
                dist = np.arange(1., pred_len+1., 1., dtype=float)
                traj_ramp_x = dist
                traj_ramp_y = np.zeros_like(traj_ramp_x)
            elif desired_source == 'simfork':
                dist = np.arange(1., pred_len+1., 1., dtype=float)
                traj_ramp_x = dist * np.cos(sgn * theta, dtype=float)
                traj_ramp_y = dist * np.sin(sgn * theta, dtype=float)
            elif desired_source == 'simcross':
                dist = np.arange(1., pred_len+1., 1., dtype=float)
                traj_ramp_x = dist * np.cos(sgn * theta, dtype=float)
                dist = np.concatenate([np.arange(1., pred_len // 3 + 1., 1., dtype=float), np.arange(pred_len // 3 - 1., -pred_len // 3 - 1., -1., dtype=float)])
                traj_ramp_y = dist * np.sin(sgn * theta, dtype=float)
                
            traj_ramp = np.concatenate([traj_ramp_x[:, None], traj_ramp_y[:, None]], axis=1)
            _traj = np.concatenate([traj_flat, traj_ramp], axis=0)
    
            traj_base.append(_traj)
        traj_base = np.array(traj_base)
        
        traj = np.tile(traj_base, (int(N_seqs / traj_base.shape[0]), 1, 1))
        traj += np.random.normal(loc=np.zeros_like(traj), scale=np.ones_like(traj) * std)
        
        id_seq = np.tile(np.arange(N_seqs)[:, None, None], [1, seq_len, 1])
        traj = np.concatenate([id_seq, traj], axis=2)
        
        id_frame = np.arange(N_seqs * seq_len)[:, None]
        traj = np.concatenate([id_frame, traj.reshape([N_seqs * seq_len, -1])], axis=1)

        data = pd.DataFrame(traj, columns=['frame_id', 'track_id', 'pos_x', 'pos_y'])
        data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
        data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')

        #data['frame_id'] = data['frame_id'] // 10

        data['node_type'] = 'PEDESTRIAN'
        data['node_id'] = data['track_id'].astype(str)

        data.sort_values('frame_id', inplace=True)
        
        #data['pos_x'] = data['pos_x'] - data['pos_x'].mean()
        #data['pos_y'] = data['pos_y'] - data['pos_y'].mean()

        max_timesteps = data['frame_id'].max()

        scene = Scene(timesteps=max_timesteps+1, dt=dt, name=f'{desired_source}_{data_class}', aug_func=None)

        for node_id in pd.unique(data['node_id']):

            node_df = data[data['node_id'] == node_id]

            node_values = node_df[['pos_x', 'pos_y']].values

            if node_values.shape[0] < 2:
                continue

            new_first_idx = node_df['frame_id'].iloc[0]

            x = node_values[:, 0]
            y = node_values[:, 1]
            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

            data_dict = {('position', 'x'): x,
                            ('position', 'y'): y,
                            ('velocity', 'x'): vx,
                            ('velocity', 'y'): vy,
                            ('acceleration', 'x'): ax,
                            ('acceleration', 'y'): ay}

            node_data = pd.DataFrame(data_dict, columns=data_columns)
            node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data)
            node.first_timestep = new_first_idx

            scene.nodes.append(node)
            
        env.gt_dist = np.concatenate([traj_base, np.ones_like(traj_base) * std], axis=2)
        
        print(scene)
        scenes.append(scene)
        print(f'Processed {len(scenes):.2f} scene for data class {data_class}')

        env.scenes = scenes

        if len(scenes) > 0:
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
