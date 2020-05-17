import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import collections
import gym

class PolicyNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
            # nn.ReLU(),
            # nn.Linear(32, n_actions)
        )
    
    def forward(self, x):
        output = self.fc(x)
        return F.softmax(output, dim = 1)

class REINFORCE:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.device = torch.device('cuda')

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.policy_network = PolicyNetwork(self.n_states, self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr = 1e-2)

        self.gamma = 0.99
        self.total_rewards = []
        self.best_mean_reward = None

        self.steps = 0
        self.writer = SummaryWriter(comment = self.env_name + '-myreinforce')

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        state = torch.unsqueeze(state, dim = 0)

        probs = self.policy_network(state).squeeze()
        probs = probs.detach().cpu().numpy()

        action = np.random.choice(np.arange(self.n_actions), p = probs)

        return action
    
    def get_discounted_rewards(self, rewards):
        cumulative_rewards = 0.0
        n = rewards.shape[0]
        for i in range(n - 1, -1, -1):
            cumulative_rewards += rewards[i]
            rewards[i] = cumulative_rewards
            cumulative_rewards *= self.gamma

        eps = np.finfo(np.float32).eps.item()
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        return np.array(rewards)

    def optimize(self, states, actions, rewards):
        states = torch.FloatTensor(states).to(self.device)
        probs = self.policy_network(states)
        actions = torch.tensor(actions).to(self.device)

        log_probs = torch.log(probs)
        log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        rewards = torch.tensor(self.get_discounted_rewards(rewards)).to(self.device)

        loss = -log_probs * rewards
        loss = loss.sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run_train_episode(self):
        state = self.env.reset()

        states = []
        actions = []
        rewards = []
        total_reward = 0.0
        
        start_frame = self.steps
        start_time = time.time()

        while True:
            self.steps += 1
            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            rewards.append(reward)
            states.append(state)
            actions.append(action)

            if done:
                self.total_rewards.append(total_reward)
                mean_reward = np.mean(self.total_rewards[-100:])
                if self.best_mean_reward is None or self.best_mean_reward < mean_reward:
                    torch.save(self.policy_network.state_dict(), self.env_name + '-myreinforce.pth')
                    if self.best_mean_reward is not None:
                        print("Best mean reward updated, %0.3f -> %0.3f. model saved" % (self.best_mean_reward, mean_reward))
                    self.best_mean_reward = mean_reward
                speed = (self.steps - start_frame) / (time.time() - start_time)

                self.writer.add_scalar("reward", total_reward)
                self.writer.add_scalar("mean_reward", mean_reward)
                self.writer.add_scalar("speed", speed)
                
                print("%d: Done %d games, mean reward %.3f, speed %.2f f/s" % (self.steps, len(self.total_rewards), mean_reward, 
                    speed))

                
                break
            state = next_state
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        self.optimize(states, actions, rewards)

    def train(self):
        for _ in range(4000):
            self.run_train_episode()
    
    def load_network(self):
        self.policy_network.load_state_dict(torch.load(self.env_name + '-myreinforce.pth'))

    def run_test_episode(self):
        total_reward = 0.0
        self.load_network()
        state = self.env.reset()

        while True:
            action = self.get_action(state)
            self.env.render()
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            if done:
                self.total_rewards.append(total_reward)
                print("Done %d games, reward %.3f" % (len(self.total_rewards), total_reward))
                break
            state = next_state

    def test(self):
        for _ in range(20):
            self.run_test_episode()       


if __name__ == '__main__':
    reinforce_agent = REINFORCE('CartPole-v0')
    reinforce_agent.train()