from gym_input import GymSim


class DoomSim(GymSim):

    @property
    def ACTION_DIM(self):
        #TODO: fix this outrageously hacky solution
        return self._env.action_space.sample().shape

    @property
    def state(self):
        state = self._state_preprocess(self.render(True))
        return np.expand_dims(state, 0)

    def _state_preprocess(self, state):
        # cuz image
        state = state.transpose((2,0,1))
        return state

if __name__ == '__main__':
    sim = GymSim('DoomBasic-v0', 1000)
    sim.act_sample_batch(1000, 1)
    sim.dump_db('doom_1000.h5')
