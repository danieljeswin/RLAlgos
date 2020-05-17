import numpy as np
import gym
from torch.utils.tensorboard import SummaryWriter
import collections


class ValueIterationAgent:
    TEST_EPISODES = 20

    def __init__(self):
        self.env = gym.make('FrozenLake-v0')
        self.test_env = gym.make('FrozenLake-v0')
        self.gamma = 0.9
        self.state = self.env.reset()

        self.rewards = collections.defaultdict(float)
        self.transitions = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

        self.writer = SummaryWriter(comment = "value-iteration")

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)

            self.rewards[(self.state, action, next_state)] = reward
            self.transitions[(self.state, action)][next_state] += 1
            self.state = self.env.reset() if done else next_state
    
    def calc_action_value(self, state, action):
        total = sum(self.transitions[(state, action)].values())
        action_value = 0.0
        
        for next_state, count in self.transitions[(state, action)].items():
            reward = self.rewards[(state, action, next_state)] 
            value = self.values[next_state]
            action_value += (count / total) * (reward + self.gamma * value)
        return action_value

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self):
        state = self.test_env.reset()
        total_reward = 0.0

        while True:
            action = self.select_action(state)
            next_state, reward, done, _ = self.test_env.step(action)
            total_reward += reward
            self.rewards[(state, action, next_state)] = reward
            self.transitions[(state, action)][next_state] += 1

            if done:
                break
            state = next_state
        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            action_values = [self.calc_action_value(state, action) for action in range(self.env.action_space.n)]
            self.values[state] = max(action_values)

    def train_run_agent(self):
        iter_num = 0.0
        best_reward = 0.0
        while True:
            iter_num += 1
            self.play_n_random_steps(100)
            self.value_iteration()
            reward = 0.0
            for _ in range(ValueIterationAgent.TEST_EPISODES):
                reward += self.play_episode()
            reward /= ValueIterationAgent.TEST_EPISODES
            self.writer.add_scalar("reward", reward, iter_num)

            if reward > best_reward:
                print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
                best_reward = reward
            
            if reward > 0.8:
                print("Solved in %d iterations" % (iter_num))
                break
        self.writer.close()

if __name__ == "__main__":
    agent = ValueIterationAgent()
    agent.train_run_agent()

    
        