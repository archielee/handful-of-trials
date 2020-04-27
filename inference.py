from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from dotmap import DotMap
from dmbrl.config import create_config
from dmbrl.misc.DotmapUtils import get_required_argument
from scipy.io import loadmat


def main(env, ctrl_type, ctrl_args, overrides, model_dir, logdir):
    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})

    overrides.append(["ctrl_cfg.prop_cfg.model_init_cfg.model_dir", model_dir])
    overrides.append(["ctrl_cfg.prop_cfg.model_init_cfg.load_model", "True"])
    overrides.append(["ctrl_cfg.prop_cfg.model_pretrained", "True"])
    overrides.append(["exp_cfg.exp_cfg.ninit_rollouts", "0"])
    overrides.append(["exp_cfg.exp_cfg.ntrain_iters", "1"])
    overrides.append(["exp_cfg.log_cfg.nrecord", "1"])

    cfg_dotmap = create_config(env, ctrl_type, ctrl_args, overrides, logdir)

    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    with tf.Session(config=cfg) as sess:
        model_init_cfg = cfg_dotmap.ctrl_cfg.prop_cfg.model_init_cfg
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model", num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
            sess=sess, load_model=model_init_cfg.get("load_model", False),
            model_dir=model_init_cfg.get("model_dir", None)
        ))
        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})

        T = 20
        dt = 0.1
        data_path = os.path.join(model_dir, "train_iter50", "logs.mat")
        data = loadmat(data_path)
        acs = data["actions"][-10:,:T].astype(np.float32)
        obs = data["observations"][-10:,:T+1].astype(np.float32)
        x_init = obs[:,0]

        traj_means = np.zeros((obs.shape[0], T, obs.shape[2]))
        traj_vars = np.zeros((obs.shape[0], T, obs.shape[2]))
        for i in range(T):
            if isinstance(x_init, np.ndarray):
                x_init = tf.convert_to_tensor(x_init) 
            ac = tf.convert_to_tensor(acs[:,i])
            ob = tf.concat([tf.expand_dims(tf.sin(x_init[:,0]), axis=-1), tf.expand_dims(tf.cos(x_init[:,0]), axis=-1), x_init[:,1:]], axis=-1)
            inputs = tf.concat([ob, ac], axis=-1)
            means, variances = model.create_prediction_tensors(inputs)
            # pitch = x_init[:,0] + means[:,0] * dt
            # q = means[:,0]
            # x = x_init[:,2] + means[:,1] * dt
            # z = x_init[:,3] + means[:,2] * dt
            # dx = means[:,1]
            # dz = means[:,2]
            # x_init = tf.stack([pitch, q, x, z, dx, dz], axis=-1)
            x_init = means
            mean = sess.run(x_init)
            variance = sess.run(variances)
            traj_means[:,i,:] = mean
            # pos_idxs = [0,2,3]
            # rate_idxs = [1,4,5]
            # traj_vars[:,i,rate_idxs] = variance
            # if i > 0:
            #     traj_vars[:,i,pos_idxs] = variance * dt**2 + traj_vars[:,i-1,pos_idxs]
            # else:
            #     traj_vars[:,i,pos_idxs] = variance * dt**2
            traj_vars[:,i,:] = variance
        gt = obs[:,1:]

        file_path = os.path.join(os.getcwd(), "data_pos.npz")
        np.savez(file_path, traj_means=traj_means, traj_vars=traj_vars, gt=gt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, required=True)
    parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2, default=[])
    parser.add_argument('-o', '--override', action='append', nargs=2, default=[])
    parser.add_argument('-model-dir', type=str, required=True)
    parser.add_argument('-logdir', type=str, required=True)
    args = parser.parse_args()

    main(args.env, "MPC", args.ctrl_arg, args.override, args.model_dir, args.logdir)