from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    file_path = os.path.join(os.getcwd(), "data", "data_pets_pos.npz")
    data_pets_pos = np.load(file_path)

    file_path = os.path.join(os.getcwd(), "data", "data_pets_rates.npz")
    data_pets_rate = np.load(file_path)

    file_path = os.path.join(os.getcwd(), "data", "pred_pets_rates.npz")
    pred_pets_rate = np.load(file_path)

    file_path = os.path.join(os.getcwd(), "data", "pred_pets_pos.npz")
    pred_pets_pos = np.load(file_path)

    file_path = os.path.join(os.getcwd(), "data", "data_pilco.npz")
    data_pilco = np.load(file_path)

    file_path = os.path.join(os.getcwd(), "data", "data_gp.npz")
    data_gp = np.load(file_path)

    file_path = os.path.join(os.getcwd(), "data", "pred_gp.npz")
    pred_gp = np.load(file_path)

    file_path = os.path.join(os.getcwd(), "data", "data_mpc.npz")
    data_mpc = np.load(file_path)

    file_path = os.path.join(os.getcwd(), "data", "rewards.npz")
    rewards = np.load(file_path)

    gt = pred_pets_pos["gt"]
    pets_means_pos = pred_pets_pos["traj_means"]
    pets_vars_pos = pred_pets_pos["traj_vars"]
    pets_means_rate = pred_pets_rate["traj_means"]
    pets_vars_rate = pred_pets_rate["traj_vars"]
    gp_means = pred_gp["traj_means"]
    gp_vars = pred_gp["traj_vars"]

    traj_mpc = data_mpc["obs"]
    traj_pets_pos = data_pets_pos["obs"]
    traj_pets_rate = data_pets_rate["obs"]
    traj_pilco = data_pilco["data"][:,:,:6]
    traj_gp = data_gp["obs"]

    rew_pets_pos = rewards["rew_pets_pos"]
    rew_pets_rate = rewards["rew_pets_rate"]
    rew_pilco = rewards["rew_pilco"]
    rew_gp = -1 * data_gp["rew"][1:]

    T = 20

    fig = plt.figure(figsize=(5,5))
    plt.scatter([5], [10], color='r', marker='*', label="Goal")
    plt.plot(traj_mpc[:,2], traj_mpc[:,3], label="MPC w/ perfect model")
    plt.plot(traj_pets_pos[-1,:,2], traj_pets_pos[-1,:,3], label="PETS")
    plt.plot(traj_pets_rate[-1,:,2], traj_pets_rate[-1,:,3], label="PETS (rates only)")
    plt.plot(traj_pilco[-1,:,2], traj_pilco[-1,:,3], label="PILCO")
    plt.plot(traj_gp[-1,:,2], traj_gp[-1,:,3], label="GP+MPC")
    plt.xlim([-2.5, 7.5]); plt.ylim([5, 15])
    plt.xlabel("x (m)"); plt.ylabel("z (m)")
    plt.legend()

    pets_dist_pos = np.sqrt(np.sum(np.square(traj_pets_pos[-1,-1,2:4] - np.array([5, 10])), axis=-1))
    pets_dist_rate = np.sqrt(np.sum(np.square(traj_pets_rate[-1,-1,2:4] - np.array([5, 10])), axis=-1))
    gp_dist = np.sqrt(np.sum(np.square(traj_gp[-1,-1,2:4] - np.array([5, 10])), axis=-1))
    print(pets_dist_pos)
    print(pets_dist_rate)
    print(gp_dist)


    t = np.linspace(1, 50, num=50)
    fig = plt.figure()
    plt.hlines(171.16021323669702, 1, 50, colors='k', linestyles="dashed", label="MPC w/ perfect model")
    plt.plot(t, rew_pets_pos, label="PETS (learned state)")
    plt.plot(t, rew_pets_rate, label="PETS (learned rates)")
    plt.plot(t, rew_gp, label="GP+MPC")
    plt.xlabel("# of training episodes"); plt.ylabel("Cost")
    plt.ylim([0, 18000])
    plt.legend()

    err_pos = pets_means_pos - gt
    rmse_pos = np.average(np.sqrt(np.average(np.square(err_pos), axis=0)), axis=-1)
    print("full")
    print(rmse_pos[0])
    print(rmse_pos[4])
    print(rmse_pos[9])
    print(rmse_pos[14])
    print(rmse_pos[19])

    err_rate = pets_means_rate - gt
    rmse_rate = np.average(np.sqrt(np.average(np.square(err_rate), axis=0)), axis=-1)
    print("rate")
    print(rmse_rate[0])
    print(rmse_rate[4])
    print(rmse_rate[9])
    print(rmse_rate[14])
    print(rmse_rate[19])

    err_gp = gp_means - gt
    rmse_gp = np.average(np.sqrt(np.average(np.square(err_gp), axis=0)), axis=-1)
    print("gp")
    print(rmse_gp[0])
    print(rmse_gp[4])
    print(rmse_gp[9])
    print(rmse_gp[14])
    print(rmse_gp[19])

    gt[:,:,[1,2,3]] = gt[:,:,[2,3,1]]
    pets_means_pos[:,:,[1,2,3]] = pets_means_pos[:,:,[2,3,1]]
    pets_means_rate[:,:,[1,2,3]] = pets_means_rate[:,:,[2,3,1]]
    pets_vars_pos[:,:,[1,2,3]] = pets_vars_pos[:,:,[2,3,1]]
    pets_vars_rate[:,:,[1,2,3]] = pets_vars_rate[:,:,[2,3,1]]

    n_state = pets_means_pos.shape[-1] // 2
    t = np.linspace(start=0.1, stop=0.1*T, num=T)
    fig, axs = plt.subplots(2, 3, constrained_layout=True, figsize=(15, 5))
    labels = ["pitch", "x", "z", "q", "x vel", "z vel"]
    for i in range(n_state):
        axs[0,i].plot(t, gt[-1,:,i], label="Ground truth", color='r')
        axs[0,i].plot(t, pets_means_pos[-1,:,i], label="NN", color='b')
        axs[0,i].fill_between(t, pets_means_pos[-1,:,i]+3*np.sqrt(pets_vars_pos[-1,:,i]), pets_means_pos[-1,:,i]-3*np.sqrt(pets_vars_pos[-1,:,i]), facecolor='b', alpha=0.5)
        axs[0,i].plot(t, pets_means_rate[-1,:,i], label="NN (learned rates)", color='m')
        axs[0,i].fill_between(t, pets_means_rate[-1,:,i]+3*np.sqrt(pets_vars_rate[-1,:,i]), pets_means_rate[-1,:,i]-3*np.sqrt(pets_vars_rate[-1,:,i]), facecolor='m', alpha=0.5)
        axs[0,i].plot(t, gp_means[-1,:,i], label="GP", color='g')
        axs[0,i].fill_between(t, gp_means[-1,:,i]+3*np.sqrt(gp_vars[-1,:,i]), gp_means[-1,:,i]-3*np.sqrt(gp_vars[-1,:,i]), facecolor='g', alpha=0.5)
        axs[0,i].set_ylabel(labels[i]); axs[0,i].set_xlabel("t (s)")
        axs[0,i].legend()
        axs[1,i].plot(t, gt[-1,:,i+n_state], label="Ground truth", color='r')
        axs[1,i].plot(t, pets_means_pos[-1,:,i+n_state], label="NN prediction", color='b')
        axs[1,i].fill_between(t, pets_means_pos[-1,:,i+n_state]+3*np.sqrt(pets_vars_pos[-1,:,i+n_state]), pets_means_pos[-1,:,i+n_state]-3*np.sqrt(pets_vars_pos[-1,:,i+n_state]), facecolor='b', alpha=0.5)
        axs[1,i].plot(t, pets_means_rate[-1,:,i+n_state], label="NN (learned rates)", color='m')
        axs[1,i].fill_between(t, pets_means_rate[-1,:,i+n_state]+3*np.sqrt(pets_vars_rate[-1,:,i+n_state]), pets_means_rate[-1,:,i+n_state]-3*np.sqrt(pets_vars_rate[-1,:,i+n_state]), facecolor='m', alpha=0.5)
        axs[1,i].plot(t, pets_means_pos[-1,:,i+n_state], label="GP prediction", color='g')
        axs[1,i].fill_between(t, pets_means_pos[-1,:,i+n_state]+3*np.sqrt(gp_vars[-1,:,i+n_state]), pets_means_pos[-1,:,i+n_state]-3*np.sqrt(gp_vars[-1,:,i+n_state]), facecolor='g', alpha=0.5)
        axs[1,i].set_ylabel(labels[i+n_state]); axs[1,i].set_xlabel("t (s)")
        axs[1,i].legend()

    plt.show()