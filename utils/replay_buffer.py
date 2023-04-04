import random
import numpy as np
from collections import deque, namedtuple


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        batch = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = list(map(np.array, list(zip(*batch))))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
