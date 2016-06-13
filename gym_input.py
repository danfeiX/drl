from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import collections
import numpy as np
import h5py
from IPython import embed


class GymSim:

    def __init__(self, sim_name, max_sample=0, seed=0):
        self._env = gym.make(sim_name)
        self._sim_name = sim_name
        self._env.seed(seed)

        self._iter_count = 0
        self._episode_count = 0

        if max_sample <= 0:
            self._sample_base = []
        else:
            self._sample_base = collections.deque(maxlen=max_sample)

        self._episode_num_pos = 0 # num positive samples from the current episode
        self._episode_num_neg = 0 # num negative samples from the current episode


    @property
    def state(self):
        state = self._state_preprocess(self._env.state)
        return np.expand_dims(state, 0)


    def _state_preprocess(self, state):
        return np.array(state)

    @property
    def ACTION_DIM(self):
        return self._env.action_space.n


    @property
    def INPUT_DIM(self):
        return self._env.observation_space.shape[0]


    def reset(self):
        self._episode_num_pos = 0
        self._episode_num_neg = 0
        self._episode_count += 1
        self._env.reset()
        print(self._episode_count)


    def _act(self, action=None):
        """
        A wrapper function for acting in the gym
        environment.
        """
        state = self.state

        if action is None:
            action = self._env.action_space.sample()

        _, reward, done, info = self._env.step(action)
        observation = self.state

        return [state, observation, reward, done, action, info]


    def act_sample_batch(self, num_sample, neg_ratio=0):
        for _ in xrange(num_sample):
            self.act_sample_once(neg_ratio=neg_ratio, append_db=True)


    def act_sample_once(self, action=None, neg_ratio=0, append_db=False):
        state = self.state
        sampling_neg = False

        state, observation, reward, done, action, info = self._act(action)

        if done:
            # if not sampling negative and new episode
            if neg_ratio * self._episode_num_pos <= self._episode_num_neg:
                self.reset()
                state, observation, reward, done, action, info = self._act(action)
            #if sampling negative
            else:
                sampling_neg = True

        if sampling_neg:
            self._episode_num_neg += 1
        else:
            self._episode_num_pos += 1

        if append_db:
            self.append_db([state, action, reward, observation])

        self._iter_count += 1


    def act_demo(self, action, vis=True):
        _, _, _, done, _, _ = self._act(action)
        if done:
            self.reset()
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


    def render(self, return_im=False):
        if return_im:
            return self._env.render('rgb_array')
        else:
            self._env.render()


    def dump_db(self, fn):
        h5 = h5py.File(fn, 'w')

        state = np.vstack([sample[0] for sample in self._sample_base])
        action = np.array([sample[1] for sample in self._sample_base])
        reward = np.array([sample[2] for sample in self._sample_base])
        observ = np.vstack([sample[3] for sample in self._sample_base])

        h5.create_dataset("state", data=state)
        h5.create_dataset("action", data=action)
        h5.create_dataset("reward", data=reward)
        h5.create_dataset("observ", data=observ)
        h5.close()


if __name__ == '__main__':
    sim = GymSim('CartPole-v0', 1000)
    sim.act_sample_batch(1000, 1)
    sim.dump_db('cartpole_1000.h5')
