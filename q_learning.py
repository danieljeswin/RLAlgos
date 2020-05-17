import gym
import numpy as np
import collections
from torch.utils.tensorboard import SummaryWriter

class QLearningAgent:
    def __init__(self):
        self.env = gym.make('FrozenLake-v0')
        self.gamma = 0.9
        self.learning_rate = 0.2
        self.state = self.env.reset()

        self.test_env = gym.make('FrozenLake-v0')
        self.values = collections.defaultdict(float)
        self.writer = SummaryWriter(comment = 'q-learning')

    def sample_env(self):
        action = self.env.action_space.sample()
        state = self.state
        next_state, reward, done, _ = self.env.step(action)
        self.state = self.env.reset() if done else next_state

        return (state, action, next_state, reward)

    def best_value_action(self, state):
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            value = self.values[(state, action)]
            if best_value is None or best_value < value:
                best_value = value
                best_action = action
        
        return best_action, best_value

    def value_update(self, state, action, next_state, reward):
        _, best_value = self.best_value_action(next_state)
        Q_target = reward + self.gamma * best_value
        Q_current = self.values[(state, action)]

        self.values[(state, action)] = self.learning_rate * Q_target + (1 - self.learning_rate) * Q_current

    def play_episode(self):
        total_reward = 0.0
        state = self.test_env.reset()
        while True:
            action, _ = self.best_value_action(state)    
            next_state, reward, done, _ = self.test_env.step(action)
            total_reward += reward

            if done:
                break
            state = next_state
        return total_reward
    
    def train_run_agent(self):
        iter_num = 0
        best_reward = 0.0
        while True:
            iter_num += 1
            state, action, next_state, reward = self.sample_env()
            self.value_update(state, action, next_state, reward)

            reward = 0.0
            for _ in range(20):
                reward += self.play_episode()
            reward /= 20
            self.writer.add_scalar("reward", reward, iter_num)

            if reward > best_reward:
                print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
                best_reward = reward
            if reward > 0.80:
                print("Solved in %d iterations!" % iter_num)
                break
        self.writer.close()

if __name__ == "__main__":
    agent = QLearningAgent()
    agent.train_run_agent()
    