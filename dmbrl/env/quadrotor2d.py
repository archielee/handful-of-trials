# **********************************************************************
#
# Copyright (c) 2019, Autonomous Systems Lab
# Author: Jaeyoung Lim <jalim@student.ethz.ch>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# *************************************************************************
import gym
from gym import error, spaces, utils, logger
import numpy as np
from gym.utils import seeding

import os


class Quadrotor2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        # Quadrotor parameters
        self.m = 0.5   # mass
        self.Iy = 1e-2 # moment of inertia about y (body)
        self.kr = 1e-4 # rotational drag coefficient
        self.kt = 1e-4 # translational drag coefficient
        self.l = 0.125 # arm length

        self.dt = 0.1
        self.g = 9.81

        # Quadrotor state and reference signals
        self.x = None
        self.x_goal = np.array([0., 0., 5., 10., 0., 0.])
        self.x_dim = 6

        # Conditions to end the episode
        self.height_limit = 10.
        self.horizontal_limit = 5.
        self.tol = 1e-5
        self.steps_beyond_done = None

        # Cost function parameters
        self.Q = np.diag([3e0, 1e0, 1e0, 1e0, 1e-1, 2e0])
        self.R = 5e0 * np.eye(2)

        # Rendering
        self.viewer = None
        self.x_range = 10.

        # Actions are deviations from motor forces for hover (i.e. mg/2) (N)
        act_low_bounds = np.array([-0.96 * self.m * self.g,           # total thrust deviation (N)
                                   -1.23 * self.m * self.g * self.l]) # moment about pitch (Nm)
        act_high_bounds = np.array([1.5 * self.m * self.g,            # total thrust deviation (N)
                                    1.23 * self.m * self.g * self.l]) # moment about pitch (Nm)
        self.action_space = spaces.Box(low=act_low_bounds, high=act_high_bounds)
        # Observations are full 6D state, quadrotor bounded in 20m x 20m box
        obs_low_bounds = np.array([-np.pi, # pitch angle
                                   -3.,    # pitch rate
                                   -10.,   # x position (inertial)
                                   0.,     # z position (inertial)
                                   -5.,    # x velocity (inertial)
                                   -5.])   # z velocity (inertial)
        obs_high_bounds = np.array([np.pi - np.finfo(np.float32).eps, # pitch angle
                                    3.,                               # pitch rate
                                    10.,                              # x position (inertial)
                                    20.,							  # z position (inertial)
                                    5.,  							  # x velocity (inertial)
                                    5.])							  # z velocity (inertial)
        self.observation_space = spaces.Box(low=obs_low_bounds, high=obs_high_bounds)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_goal(self, goal):
        assert type(goal) == np.ndarray
        assert goal.shape == (self.x_dim,)
        self.x_goal = goal

    def step(self, u):
        # Correction to nominal input (hover)
        T = u[0] + self.m * self.g # total thrust (N)
        My = u[1]                  # moment about pitch (Nm)

        # 2D quadrotor dynamics model following Freddi, Lanzon, and Longhi, IFAC 2011
        x_cur = self.x
        x_dot = np.empty(x_cur.shape)
        x_dot[0] = x_cur[1]
        x_dot[1] = -self.kr/self.Iy*x_cur[1] + 1/self.Iy*My
        x_dot[2] = x_cur[4]
        x_dot[3] = x_cur[5]
        x_dot[4] = -self.kt/self.m*x_cur[4] + 1/self.m*np.sin(x_cur[0])*T
        x_dot[5] = -self.kt/self.m*x_cur[5] + 1/self.m*np.cos(x_cur[0])*T - self.g

        x_next = x_cur + x_dot * self.dt
        # Wrap angle
        if x_next[0] > np.pi:
            x_next[0] -= 2*np.pi
        elif x_next[0] < -np.pi:
            x_next[0] += 2*np.pi

        self.x = x_next
        e = self.x - self.x_goal
        # Correct angle error
        if e[0] > np.pi:
            e[0] -= 2*np.pi
        elif e[0] < -np.pi:
            e[0] += 2*np.pi

        # NOTE: Don't think PETS supports early termination
        # # Check failure conditions to end episode
        # hit_ground = self.state[3] <= 0.
        # too_high = self.state[3] >= self.height_limit
        # too_far = not (-self.horizontal_limit <= self.state[2] <= self.horizontal_limit)
        # reached_goal = np.linalg.norm(e) < self.tol
        # done = hit_ground or too_high or too_far or reached_goal
        # if not done:
        #     # Standard quadratic cost function from ILQR
        #     reward = -0.5 * (e.T @ self.Q @ e + u.T @ self.R @ u)
        # elif self.steps_beyond_done is None:
        #     self.steps_beyond_done = 0
        #     # Huge cost for crashing
        #     if hit_ground:
        #         reward = -1e9
        #     # Terminal cost from ILQR
        #     elif reached_goal:
        #         reward = -0.5 * (e.T @ self.Q @ e) 
        #     # Cost for leaving boundary
        #     else:
        #         reward = -1e3
        # else:
        #     if self.steps_beyond_done == 0:
        #         logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
        #     self.steps_beyond_done += 1
        #     reward = 0.0

        done = False
        reward = -0.5 * (e.T @ self.Q @ e + u.T @ self.R @ u)
        # if self.state[3] <= 0.:
        #     reward += -1e4

        return self.x, reward, done, {}

    def reset(self):
        print("Environment reset")
        # self.x = self.np_random.uniform(low=-0.5, high=0.5, size=self.observation_space.shape)
        self.x = np.zeros(6)
        # self.x[0] = self.np_random.uniform(low=-5*np.pi/180., high=5*np.pi/180.) # spawn at some random small pitch angle
        # self.x[2] = self.np_random.uniform(low=-2., high=2.)                     # spawn at some x location
        # self.x[3] = self.np_random.uniform(low=8., high=12.)                     # spawn at some height in the middle
        self.x[3]= 10
        return self.x

    def render(self, mode='human', close=False):
        screen_width = 800
        screen_height = 800

        world_width = self.x_range*2
        scale = screen_width/world_width
        ref_size = 5.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # Draw reference
            ref = rendering.make_circle(ref_size)
            self.reftrans = rendering.Transform()
            ref.add_attr(self.reftrans)
            ref.set_color(1,0,0)
            self.viewer.add_geom(ref)
            # Draw start
            start = rendering.make_circle(ref_size)
            self.starttrans = rendering.Transform()
            start.add_attr(self.starttrans)
            start.set_color(0,0,0)
            self.viewer.add_geom(start)
            # Draw drone
            dir_path = os.path.dirname(os.path.realpath(__file__))
            quad = rendering.Image('%s/assets/quadrotor2d.png' % dir_path, 60, 12)
            self.quadtrans = rendering.Transform()
            quad.add_attr(self.quadtrans)
            self.viewer.add_geom(quad)
            # image_data = quad.img.get_data()[:]
            # print(image_data)

        if self.x is None:
            return None

        quad_x = self.x[2]*scale+screen_width/2.0 
        quad_y = self.x[3]*scale 
        self.quadtrans.set_translation(quad_x, quad_y)
        self.quadtrans.set_rotation(-self.x[0])

        y = self.x_goal[2:4]
        ref_x = y[0]*scale+screen_width/2.0
        ref_y = y[1]*scale
        self.reftrans.set_translation(ref_x, ref_y)

        y = np.array([0, 10])
        start_x = y[0]*scale+screen_width/2.0
        start_y = y[1]*scale
        self.starttrans.set_translation(start_x, start_y)

        return self.viewer.render(return_rgb_array=(mode =='rgb_array'))
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None