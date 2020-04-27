from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from dotmap import DotMap
import gym

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC
import dmbrl.env


class Quadrotor2DConfigModule:
    ENV_NAME = "MBRLQuadrotor2D-v0"
    TASK_HORIZON = 50
    NTRAIN_ITERS = 75
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 20
    MODEL_IN, MODEL_OUT = 9, 6
    GP_NINDUCING_POINTS = 200

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2000
            },
            "CEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    # @staticmethod
    # def obs_preproc(obs):
    #     angs = obs[:,0]
    #     if isinstance(obs, np.ndarray):
    #         obs_proc = np.concatenate([np.expand_dims(np.sin(angs), axis=-1), np.expand_dims(np.cos(angs), axis=-1), obs[:,1:]], axis=-1)
    #     else:
    #         obs_proc = tf.concat([tf.expand_dims(tf.sin(angs), axis=-1), tf.expand_dims(tf.cos(angs), axis=-1), obs[:,1:]], axis=-1)
    #     return obs_proc

    # @staticmethod
    # def obs_postproc(obs, model_out):
    #     dt = 0.1
    #     pitch = obs[:,0] + model_out[:,0] * dt
    #     x = obs[:,2] + model_out[:,1] * dt
    #     z = obs[:,3] + model_out[:,2] * dt
    #     q = model_out[:,0]
    #     dx = model_out[:,1]
    #     dz = model_out[:,2]
    #     if isinstance(obs, np.ndarray):
    #         new_obs = np.stack([pitch, q, x, z, dx, dz], axis=-1)
    #     else:
    #         new_obs = tf.stack([pitch, q, x, z, dx, dz], axis=-1)
    #     return new_obs

    @staticmethod
    def targ_proc(obs, next_obs):
        rate_idxs = [1,4,5]
        return next_obs[:,rate_idxs]

    @staticmethod
    def obs_cost_fn(obs):
        Q = np.diag([3e0, 1e0, 1e0, 1e0, 1e-1, 2e0]).astype(np.float32)
        goal = np.array([0., 0., 5., 10., 0., 0.], dtype=np.float32)
        if isinstance(obs, np.ndarray):
            e = obs - goal
            cost = 0.5 * np.einsum('ij,ij->i', np.einsum('ij,jk->ik', e, Q), e)
        else:
            Q = tf.constant(Q, dtype=tf.float32)
            goal = tf.constant(goal, dtype=tf.float32)
            e = obs - goal
            cost = tf.cast(0.5 * tf.einsum('ij,ij->i', tf.einsum('ij,jk->ik', e, Q), e), tf.float32)
        return cost

    @staticmethod
    def ac_cost_fn(acs):
        R = 5e0 * np.eye(2)
        if isinstance(acs, np.ndarray):
            return 0.5 * np.einsum('ij,ij->i', np.einsum('ij,jk->ik', acs, R), acs)
        else:
            R = tf.constant(R, dtype=tf.float32)
            return 0.5 * tf.einsum('ij,ij->i', tf.einsum('ij,jk->ik', acs, R), acs)
        return 0

    def nn_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model", num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
            sess=self.SESS, load_model=model_init_cfg.get("load_model", False),
            model_dir=model_init_cfg.get("model_dir", None)
        ))
        if not model_init_cfg.get("load_model", False):
            model.add(FC(200, input_dim=self.MODEL_IN, activation="swish", weight_decay=0.000025))
            model.add(FC(200, activation="swish", weight_decay=0.00005))
            model.add(FC(200, activation="swish", weight_decay=0.000075))
            model.add(FC(200, activation="swish", weight_decay=0.000075))
            model.add(FC(self.MODEL_OUT, weight_decay=0.0001))
        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
        return model

    def gp_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model",
            kernel_class=get_required_argument(model_init_cfg, "kernel_class", "Must provide kernel class"),
            kernel_args=model_init_cfg.get("kernel_args", {}),
            num_inducing_points=get_required_argument(
                model_init_cfg, "num_inducing_points", "Must provide number of inducing points."
            ),
            sess=self.SESS
        ))
        return model


CONFIG_MODULE = Quadrotor2DConfigModule
