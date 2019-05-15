from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, os.path.join(dir_path, 'assets/ant.xml'), 5)
        utils.EzPickle.__init__(self)
        self.prev_pos = None

    def _step(self, a):
        self.prev_pos = np.copy(self.get_body_com("torso"))
        self.do_simulation(a, self.frame_skip)
        done = False
        ob = self._get_obs()

        reward = ob[0] - 0.02 * np.sum(np.square(a))
        return ob, reward, done, dict()

    def _get_obs(self):
        r = self.model.data.xmat[1]

        # Quaternion to Euler angles
        if np.abs(r[0]) < 1e-6 and np.abs(r[3]) < 1e-6:
            beta = np.pi / 2
            alpha = 0
            gamma = np.arctan2(r[1], r[4])
        else:
            beta = np.arctan2(-r[6], np.sqrt(r[0] * r[0] + r[3] * r[3]))
            alpha = np.arctan2(r[3], r[0])
            gamma = np.arctan2(r[7], r[8])
        euler = np.array([alpha, beta, gamma])

        cur_pos = self.get_body_com("torso")
        return np.concatenate([
            (cur_pos.flat[:2] - self.prev_pos.flat[:2]) / self.dt,
            cur_pos[2:3],
            euler,
            self.model.data.qpos.flat[7:],
            self.model.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.prev_pos = np.copy(self.get_body_com("torso"))
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
