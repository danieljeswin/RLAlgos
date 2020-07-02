import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import gym
import pybullet_envs

import numpy as np
import time
import collections

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


class Actor(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_states, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, n_actions),
            nn.Tanh()
        )
    
    def forward(self, states):
        output = self.model(states)
        return output

class Critic(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_states + n_actions, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )
    
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim = 1)
        output = self.model(x)
        return output


class TD3:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.device = torch.device('cuda')

        self.n_actions = self.env.action_space.shape[0]
        self.n_states = self.env.observation_space.shape[0]

        self.actor = Actor(self.n_states, self.n_actions).to(self.device)
        self.critic_1 = Critic(self.n_states, self.n_actions).to(self.device)
        self.critic_2 = Critic(self.n_states, self.n_actions).to(self.device)
        self.target_actor = Actor(self.n_states, self.n_actions).to(self.device)
        self.target_critic_1 = Critic(self.n_states, self.n_actions).to(self.device)
        self.target_critic_2 = Critic(self.n_states, self.n_actions).to(self.device)
        
        self.batch_size = 100
        self.gamma = 0.99
        self.d = 2
        self.c = 0.5
        self.learning_rate = 1e-3
        self.exploration_sigma = 0.1
        self.smoothing_sigma = 0.2
        self.tau = 0.005
        self.replay_buffer_size = int(1e6)
        self.learning_start = 10000
        self.steps = 0
        self.total_episodes = 20000

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = self.learning_rate)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr = self.learning_rate)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr = self.learning_rate)
        self.critic_loss_criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
       
        self.total_rewards = []
        self.mean_reward = None

    def save_networks(self):
        torch.save(self.actor.state_dict(), self.env_name + '-mytd3actor.pth')
        torch.save(self.critic_1.state_dict(), self.env_name + '-mytd3critic1.pth')
        torch.save(self.critic_2.state_dict(), self.env_name + '-mytd3critic2.pth')

    def load_networks(self):
        self.actor.load_state_dict(torch.load(self.env_name + '-mytd3actor.pth'))
        self.critic_1.load_state_dict(torch.load(self.env_name + '-mytd3critic1.pth'))
        self.critic_2.load_state_dict(torch.load(self.env_name + '-mytd3critic2.pth'))
        self.initialize_target_networks()

    def initialize_target_networks(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

    def update_target_networks(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def train(self):
        total_reward = 0.0
        state = self.env.reset()
        episode_num = 0
        self.writer = SummaryWriter(comment = self.env_name + '-mytd3')
        
        while episode_num < self.total_episodes:
            self.steps += 1
            if self.steps <= self.learning_start:
                action = self.env.action_space.sample()
            else:
                state_t = state.copy()
                state_t = torch.Tensor(state_t).to(self.device)
                action = self.actor(state_t).detach().cpu().numpy()
                noise = np.random.normal(loc = 0.0, scale = self.exploration_sigma, size = self.n_actions)
                action = action + noise
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            experience = Experience(state, action, next_state, reward, done)
            self.replay_buffer.append(experience)

            if self.steps > self.learning_start:
                batch = self.replay_buffer.sample(self.batch_size)
                states, actions, next_states, rewards, dones = batch

                states = torch.Tensor(states).to(self.device)
                next_states = torch.Tensor(next_states).to(self.device)
                actions = torch.Tensor(actions).to(self.device)
                rewards = torch.Tensor(rewards).to(self.device).view(-1, 1)
                terminals = torch.Tensor(dones).to(self.device).view(-1, 1)

                next_actions = self.target_actor(next_states)
                smoothing_noise = torch.randn_like(next_actions) * self.smoothing_sigma
                smoothing_noise = torch.clamp(smoothing_noise, -self.c, self.c)
                next_actions = next_actions + smoothing_noise
                next_actions = torch.clamp(next_actions, self.env.action_space.low[0], self.env.action_space.high[0])

                target_q_1 = self.target_critic_1(next_states, next_actions).detach()
                target_q_2 = self.target_critic_2(next_states, next_actions).detach()
                target_q = torch.min(target_q_1, target_q_2)

                Q_target = rewards + self.gamma * target_q * (1 - terminals)
                Q_target = Q_target.detach()
                Q_1 = self.critic_1(states, actions)
                Q_2 = self.critic_2(states, actions)
                critic_1_loss = self.critic_loss_criterion(Q_1, Q_target)
                critic_2_loss = self.critic_loss_criterion(Q_2, Q_target)

                self.critic_1_optimizer.zero_grad()
                critic_1_loss.backward()
                self.critic_1_optimizer.step()

                self.critic_2_optimizer.zero_grad()
                critic_2_loss.backward()
                self.critic_2_optimizer.step()

                if self.steps % self.d == 0:
                    actor_loss = -self.critic_1(states, self.actor(states)).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    self.update_target_networks()

            state = next_state
            if done:
                # Calculate statistics and display
                self.total_rewards.append(total_reward)
                mean_reward = np.mean([self.total_rewards[-100:]])
                print(f"Done {episode_num} games. Steps {self.steps} Episode reward {total_reward} Mean reward {mean_reward}")

                if self.mean_reward is None or self.mean_reward < mean_reward:
                    if self.mean_reward is not None:
                        print(f"New best mean {self.mean_reward} -> {mean_reward}. Network saved")
                    self.mean_reward = mean_reward
                    self.save_networks()
                self.writer.add_scalar('Reward', total_reward, self.steps)
                self.writer.add_scalar('Mean Reward', mean_reward, self.steps)
                state = self.env.reset()
                total_reward = 0.0
                episode_num += 1
        self.writer.close()

    def run_test_episode(self, episode_num):
        total_reward = 0.0
        self.env.render
        state = self.env.reset()

        while True:
            self.env.render()
            state_t = state.copy()
            state_t = torch.Tensor(state_t).to(self.device)
            action = self.actor(state_t).detach().cpu().numpy()

            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            if done:
                print(f"Episode number : {episode_num}, Reward : {total_reward}")
                break
            state = next_state

    def test(self):
        self.load_networks()
        for episode_num in range(20):
            self.run_test_episode(episode_num)
        self.env.close()


if __name__ == '__main__':
    env_name = 'HopperBulletEnv-v0'
    td3 = TD3(env_name)
    td3.train()