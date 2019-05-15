from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from dotmap import DotMap
import gym

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC



class AntConfigModule:
    ENV_NAME = "MBRLAnt-v0"
    TASK_HORIZON = 1000
    NTRAIN_ITERS = 100
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 30

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        cfg = tf.ConfigProto()
        # cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2500
            },
            "CEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    @staticmethod
    def obs_preproc(obs):
        euler = obs[:, 3:6]
        if isinstance(obs, np.ndarray):
            return np.concatenate([obs[:, 2:3], np.sin(euler), np.cos(euler), obs[:, 6:]], axis=1)
        else:
            return tf.concat([obs[:, 2:3], tf.sin(euler), tf.cos(euler), obs[:, 6:]], axis=1)

    @staticmethod
    def obs_postproc(obs, pred):
        if isinstance(obs, np.ndarray):
            return np.concatenate([pred[:, :2], obs[:, 2:] + pred[:, 2:]], axis=1)
        else:
            return tf.concat([pred[:, :2], obs[:, 2:] + pred[:, 2:]], axis=1)

    @staticmethod
    def targ_proc(obs, next_obs):
        return np.concatenate([next_obs[:, :2], next_obs[:, 2:] - obs[:, 2:]], axis=1)

    @staticmethod
    def obs_cost_fn(obs):
        return -obs[:, 0]

    @staticmethod
    def ac_cost_fn(acs):
        if isinstance(acs, np.ndarray):
            return 0.02 * np.sum(np.square(acs), axis=1)
        else:
            return 0.02 * tf.reduce_sum(tf.square(acs), axis=1)

    def nn_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model", num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
            sess=self.SESS, load_model=model_init_cfg.get("load_model", False),
            model_dir=model_init_cfg.get("model_dir", None)
        ))
        if not model_init_cfg.get("load_model", False):
            model.add(FC(200, input_dim=37, activation='swish', weight_decay=0.000025))
            model.add(FC(200, activation='swish', weight_decay=0.00005))
            model.add(FC(200, activation='swish', weight_decay=0.000075))
            model.add(FC(200, activation='swish', weight_decay=0.000075))
            model.add(FC(200, activation='swish', weight_decay=0.000075))
            model.add(FC(28, weight_decay=0.0001))
        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
        return model


CONFIG_MODULE = AntConfigModule
