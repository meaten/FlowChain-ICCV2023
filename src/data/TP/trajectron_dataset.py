from torch.utils import data
import numpy as np
from data.TP.preprocessing import get_node_timestep_data


hypers = {
    'state_p': {'PEDESTRIAN': {'position': ['x', 'y']}},
    'state_v': {'PEDESTRIAN': {'velocity': ['x', 'y']}},
    'state_a': {'PEDESTRIAN': {'acceleration': ['x', 'y']}},
    'state_pva': {
        'PEDESTRIAN': {
        'position': ['x', 'y'],
        'velocity': ['x', 'y'],
        'acceleration': ['x', 'y']
        }
    },
    'batch_size': 256,
    'grad_clip': 1.0,
    'learning_rate_style': 'exp',
    'min_learning_rate': 1e-05,
    'learning_decay_rate': 0.9999,
    'prediction_horizon': 12,
    'minimum_history_length': 1,
    'maximum_history_length': 7,
    'map_encoder':
        {'PEDESTRIAN':
            {'heading_state_index': 6,
             'patch_size': [50, 10, 50, 90],
             'map_channels': 3,
             'hidden_channels': [10, 20, 10, 1],
             'output_size': 32,
             'masks': [5, 5, 5, 5],
             'strides': [1, 1, 1, 1],
             'dropout': 0.5
            }
        },
    'k': 1,
    'k_eval': 25,
    'kl_min': 0.07,
    'kl_weight': 100.0,
    'kl_weight_start': 0,
    'kl_decay_rate': 0.99995,
    'kl_crossover': 400,
    'kl_sigmoid_divisor': 4,
    'rnn_kwargs':
        {'dropout_keep_prob': 0.75},
    'MLP_dropout_keep_prob': 0.9,
    'enc_rnn_dim_edge': 128,
    'enc_rnn_dim_edge_influence': 128,
    'enc_rnn_dim_history': 128,
    'enc_rnn_dim_future': 128,
    'dec_rnn_dim': 128,
    'q_z_xy_MLP_dims': None,
    'p_z_x_MLP_dims': 32,
    'GMM_components': 1,
    'log_p_yt_xz_max': 6,
    'N': 1,
    'tau_init': 2.0,
    'tau_final': 0.05,
    'tau_decay_rate': 0.997,
    'use_z_logit_clipping': True,
    'z_logit_clip_start': 0.05,
    'z_logit_clip_final': 5.0,
    'z_logit_clip_crossover': 300,
    'z_logit_clip_divisor': 5,
    'dynamic':
        {'PEDESTRIAN':
            {'name': 'SingleIntegrator',
             'distribution': False,
             'limits': {}
            }
        },
    'pred_state': {'PEDESTRIAN': {'velocity': ['x', 'y']}},
    'log_histograms': False,
    'dynamic_edges': 'yes',
    'edge_state_combine_method': 'sum',
    'edge_influence_combine_method': 'attention',
    'edge_addition_filter': [0.25, 0.5, 0.75, 1.0],
    'edge_removal_filter': [1.0, 0.0],
    'offline_scene_graph': 'yes',
    'incl_robot_node': False,
    'node_freq_mult_train': False,
    'node_freq_mult_eval': False,
    'scene_freq_mult_train': False,
    'scene_freq_mult_eval': False,
    'scene_freq_mult_viz': False,
    'edge_encoding': True,
    'use_map_encoding': False,
    'augment': True,
    'override_attention_radius': [],
    'learning_rate': 0.01,
    'npl_rate': 0.8,
    'K': 80,
    'tao': 0.4
}


class EnvironmentDataset(object):
    def __init__(self, env, state, pred_state, node_freq_mult, scene_freq_mult, hyperparams, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']
        self.node_type_datasets = list()
        self._augment = False
        for node_type in env.NodeType:
            if node_type not in hyperparams['pred_state']:
                continue
            self.node_type_datasets.append(NodeTypeDataset(env, node_type, state, pred_state, node_freq_mult,
                                                           scene_freq_mult, hyperparams, **kwargs))
        
    @property
    def augment(self):
        return self._augment

    @augment.setter
    def augment(self, value):
        self._augment = value
        for node_type_dataset in self.node_type_datasets:
            node_type_dataset.augment = value

    def __iter__(self):
        return iter(self.node_type_datasets)


class NodeTypeDataset(data.Dataset):
    def __init__(self, env, node_type, state, pred_state, node_freq_mult,
                 scene_freq_mult, hyperparams, augment=False, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']
        
        self.augment = augment
        
        self.node_type = node_type
        self.index = self.index_env(node_freq_mult, scene_freq_mult, **kwargs)
        self.len = len(self.index)
        self.edge_types = [edge_type for edge_type in env.get_edge_types() if edge_type[0] is node_type]    
    
    def index_env(self, node_freq_mult, scene_freq_mult, **kwargs):
        index = list()
        for scene in self.env.scenes:
            present_node_dict = scene.present_nodes(np.arange(0, scene.timesteps), type=self.node_type, **kwargs)
            for t, nodes in present_node_dict.items():
                for node in nodes:
                    index += [(scene, t, node)] *\
                             (scene.frequency_multiplier if scene_freq_mult else 1) *\
                             (node.frequency_multiplier if node_freq_mult else 1)
            
        return index

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        (scene, t, node) = self.index[i]

        if self.augment:
            scene = scene.augment()
            node = scene.get_node_by_id(node.id)
        return get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
                                      self.edge_types, self.max_ht, self.max_ft, self.hyperparams)
