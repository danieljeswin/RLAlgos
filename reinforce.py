import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import collections
import gym
from torch.distributions.categorical import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_states, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_actions)
        )
    
    def forward(self, x):
        output = self.fc(x)
        return output

class Reinforce:
    def __init__(self, env_name):
        
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.n_actions = self.env.action_space.n
        self.n_states = self.env.observation_space.shape[0]
        self.device = torch.device('cuda')

        self.learning_rate = 7e-4
        self.gamma = 0.99
        self.max_episode_length = 500
        self.total_episodes = 3000
        self.steps = 0

        self.policy = PolicyNetwork(self.n_states, self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr = self.learning_rate)
        print(self.policy)

        self.total_rewards = []
        self.mean_reward = None

    def get_discounted_rewards(self, rewards):
        for t in range(self.max_episode_length - 2, -1, -1):
            rewards[t] += self.gamma * rewards[t + 1]
        return rewards

    def save_network(self):
        torch.save(self.policy.state_dict(), self.env_name + "-myreinforce.pth")

    def load_network(self):
        self.policy.load_state_dict(torch.load(self.env_name + "-myreinforce.pth"))

    def run_train_episode(self, episode_num):
        state = self.env.reset()

        ## Initialize the array for logprob, rewards
        rewards = np.zeros(self.max_episode_length)
        logprob = torch.zeros((self.max_episode_length)).float().to(self.device)
        total_reward = 0

        for step in range(self.max_episode_length):
            self.steps += 1

            ## Logic to select action and get logprob
            state_t = state.copy()
            state_t = torch.Tensor(state_t).to(self.device)
            logits = self.policy(state_t)
            distribution = Categorical(logits = logits)
            action = distribution.sample()
            logprob[step] = distribution.log_prob(action)
            action = action.item()

            next_state, rewards[step], done, _ = self.env.step(action)
            total_reward += rewards[step]

            if done: 
                break

            state = next_state
        
        ## Calculate the loss and update the policy
        discounted_rewards = self.get_discounted_rewards(rewards)
        discounted_rewards = torch.Tensor(discounted_rewards).to(self.device)
        loss = -logprob * discounted_rewards
        loss = loss.sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Calculate statistics and display
        self.total_rewards.append(total_reward)
        mean_reward = np.mean([self.total_rewards[-100:]])
        print(f"Done {episode_num} games. Steps {self.steps} Episode reward {total_reward} Mean reward {mean_reward}")

        if self.mean_reward is None or self.mean_reward < mean_reward:
            if self.mean_reward is not None:
                print(f"New best mean {self.mean_reward} -> {mean_reward}. Network saved")
            self.mean_reward = mean_reward
            self.save_network()
        self.writer.add_scalar('Reward', total_reward, self.steps)
        self.writer.add_scalar('Mean Reward', mean_reward, self.steps)
            

    def train(self):
        self.writer = SummaryWriter(comment = self.env_name + '-myreinforce')
        for episode_num in range(self.total_episodes):
            self.run_train_episode(episode_num)
        self.writer.close()

    
    def run_test_episode(self, episode_num):
        state = self.env.reset()
        total_reward = 0
        while True:
            self.env.render()
            # Select action
            state_t = state.copy()
            state_t = torch.Tensor(state_t).to(self.device)
            logits = self.policy(state_t)
            action = torch.argmax(logits).item()

            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            if done:
                print(f"Episode number : {episode_num}, Reward : {total_reward}")
                break
            state = next_state

    def test(self):
        self.load_network()
        for episode_num in range(20):
            self.run_test_episode(episode_num)
        self.env.close()

if __name__ == '__main__':
    env_name = 'CartPole-v1'
    reinforce = Reinforce(env_name)
    reinforce.test()