from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gym
import collections
import numpy as np
from IPython import embed

FLAGS = tf.app.flags.FLAGS


class GymSim:
    def __init__(self, sim_name, db_size):
        self._env = gym.make(sim_name)
        self._env.reset()
        self._sim_name = sim_name

        self._observation_dim = self._env.observation_space.shape[0]
        self._action_dim = self._env.action_space.n
        self._iter_count = 0
        self._episode_count = 0
        self._sample_base = collections.deque(maxlen=db_size)
        self._episode_num_pos = 0 # num positive samples from the current episode
        self._episode_num_neg = 0 # num positive samples from the current episode

    @property
    def state(self):
        return np.array(self._env.state)

    def hard_done(self):
        s = self.state
        if self._sim_name == 'CartPole-v0':
            return s[0] < -2.4 or s[0] > 2.4
        return False

    def reset(self):
        self._episode_num_pos = 0
        self._episode_num_neg = 0
        self._episode_count += 1
        self._env.reset()
        print(self._episode_count)

    def act(self, action=None, neg_ratio=0, append_db=False, use_hard_done=False):
        state = self.state
        sampling_neg = False
        if action is None:
            action = self._env.action_space.sample()

        observation, reward, done, info = self._env.step(action)
        print(reward)

        if done:
            # if not sampling negative and new episode
            if (use_hard_done and self.hard_done()) or \
                neg_ratio * self._episode_num_pos <= self._episode_num_neg:
                self.reset()
                state = self.state
                observation, reward, done, info = self._env.step(action)
            #if sampling negative
            else:
                sampling_neg = True

        if sampling_neg:
            self._episode_num_neg += 1
        else:
            self._episode_num_pos += 1

        self._iter_count += 1
        if append_db:
            self.append_db([np.array(state).copy(),
                            action, reward,
                            np.array(observation).copy()])

    def act_demo(self, action, vis=True):
        _, _, done, _ = self._env.step(action)
        if done:
            self._env.reset()
        self.render()
        return done

    def append_db(self, entry):
        self._sample_base.append(entry)

    def print_stats(self):
        pos = 0; neg = 0
        for s in self._sample_base:
            if s[2] > 0:
                pos += 1
            else:
                neg += 1
        print('num_pos=%i, num_neg=%i, episode=%i'
              % (pos, neg, self._episode_count))

    @property
    def NUM_ACTION(self):
        return self._action_dim

    @property
    def INPUT_DIM(self):
        return self._observation_dim

    def next_batch(self, batch_size):
        #s, a, r, s'
        inds = np.random.choice(len(self._sample_base), batch_size)

        state_batch  = np.vstack([self._sample_base[ind][0] for ind in inds])
        action_batch = np.array([self._sample_base[ind][1] for ind in inds])
        reward_batch = np.array([self._sample_base[ind][2] for ind in inds])
        observ_batch = np.vstack([self._sample_base[ind][3] for ind in inds])

        return state_batch, action_batch, reward_batch, observ_batch

    def feed_batch(self, state_pl, action_pl, reward_pl, observ_pl, batch_size):
        s, a, r, o = self.next_batch(batch_size)
        return {state_pl: s,
                action_pl: a,
                reward_pl: r,
                observ_pl: o}

    def act_random(self, num_sample, neg_ratio=0):
        for _ in range(num_sample):
            self.act(neg_ratio=neg_ratio, append_db=True)

    def act_deepq(self, action_op, num_sample=1, frac_neg=0, add_db=True, render=False):
        for _ in range(num_sample):
            state = self.state
            action = action_op(state)
            self.act(action, frac_neg, add_db)
            if render:
                self.render()

    def render(self):
        self._env.render()
