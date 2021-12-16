# inspired from OpenAI baselines: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

import numpy as np
import torch as t


class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.curr_size = 0

    def __len__(self):
        return len(self._storage)

    def add(self, exp):
        # exp = (state, action, reward, nxt_state, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(exp)
        else:
            self._storage[self._next_idx] = exp
        self._next_idx = (self._next_idx + 1) % self._maxsize
        self.curr_size += 1

    def _encode_sample(self, idxes):
        states, actions, rewards, nxt_states, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            state, action, reward, nxt_state, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            nxt_states.append(nxt_state)
            dones.append(done)
        return t.stack(states), t.stack(actions), t.stack(rewards), t.stack(nxt_states), t.stack(dones)

    def sample(self, batch_size):
        idxes = np.random.randint(0, len(self._storage) - 1, batch_size)
        return self._encode_sample(idxes)