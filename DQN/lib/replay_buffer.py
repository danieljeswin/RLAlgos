import collections
import numpy as np

Experience = collections.namedtuple('Experience', field_names = ['state', 'action', 'next_state', 'reward', 'done'])


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen = capacity)
        self.capacity = capacity

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace = False)
        states, actions, next_states, rewards, terminals = zip(*[self.buffer[index] for index in indices])

        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards, dtype = np.float32), np.array(terminals, dtype = np.uint8)

class NStepBuffer(ReplayBuffer):
    def __init__(self, capacity, steps, gamma):
        super(NStepBuffer, self).__init__(capacity)
        self.n_step_buffer = collections.deque(maxlen = steps)
        self.steps = steps
        self.gamma = gamma
    
    def add_experience(self, e):
        self.n_step_buffer.append(e)
        done = e.done
        N = len(self.n_step_buffer)

        if (N < self.steps) and (not done):
            return

        total_reward = 0.0
        for i in range(N - 1, -1, -1):
            total_reward *= self.gamma
            total_reward += self.n_step_buffer[i].reward
        
        first_e = self.n_step_buffer[0]
        last_e = self.n_step_buffer[N - 1]

        experience = Experience(first_e.state, first_e.action, last_e.next_state, total_reward, last_e.done)
        self.buffer.append(experience)

        if done:
            self.n_step_buffer.clear()

class PriorityBuffer:
    def __init__(self, capacity, alpha = 0.6):
        self.buffer = []
        self.pos = 0
        self.capacity = capacity
        self.alpha = alpha
        self.priorities = np.zeros((capacity, ), dtype = np.float32)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        priority = 1.0
        if len(self.buffer) > 0:
            priority = self.priorities.max()
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta = 0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p = probs)
        states, actions, next_states, rewards, terminals = zip(*[self.buffer[index] for index in indices])

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards, dtype = np.float32), np.array(terminals, dtype = np.uint8), indices, np.array(weights, dtype = np.float32)

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            self.priorities[index] = priority

class NStepPriorityBuffer(PriorityBuffer):
    def __init__(self, capacity, steps, gamma, alpha = 0.6):
        super(NStepPriorityBuffer, self).__init__(capacity, alpha)
        self.n_step_buffer = collections.deque(maxlen = steps)
        self.steps = steps
        self.gamma = gamma

    def add_experience(self, e):
        self.n_step_buffer.append(e)
        done = e.done
        N = len(self.n_step_buffer)

        if (N < self.steps) and (not done):
            return

        total_reward = 0.0
        for i in range(N - 1, -1, -1):
            total_reward *= self.gamma
            total_reward += self.n_step_buffer[i].reward
        
        first_e = self.n_step_buffer[0]
        last_e = self.n_step_buffer[N - 1]

        experience = Experience(first_e.state, first_e.action, last_e.next_state, total_reward, last_e.done)
        self.buffer.append(experience)

        if done:
            self.n_step_buffer.clear()

