import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import gym

import collections
import numpy as np
import pybullet_envs

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

class OUNoise:
    def __init__(self, n_actions):
        self.theta = 0.15
        self.sigma = 0.2
        self.n_actions = n_actions
        self.mu = 0.0
        self.state = np.ones(self.n_actions) * self.mu 
        self.scale = 0.1

    def reset(self):
        self.state = np.ones(self.n_actions) * self.mu 
        
    def noise(self):
        x = self.state.copy()
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.n_actions)
        self.state = (x + dx) * self.scale
        return self.state


class Actor(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_states, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU()
        )
        self.final_layer = nn.Sequential(
            nn.Linear(300, n_actions),
            nn.Tanh()
        )
        self.model.apply(self.init_weights)
        self.final_layer.apply(self.final_layer_init_weights)
    
    @torch.no_grad()
    def final_layer_init_weights(self, m):
        value = 3e-3
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight, -value, value)

    @torch.no_grad()
    def init_weights(self, m):
        if type(m) == nn.Linear:
            f = m.weight.size()[0]
            value = 1.0 / np.sqrt(f)
            nn.init.uniform_(m.weight, -value, value)        

    def forward(self, state):
        output = self.model(state)
        output = self.final_layer(output)
        return 2 * output

class Critic(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Critic, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Linear(n_states, 400),
            nn.ReLU()
        )

        self.model = nn.Sequential(
            nn.Linear(400 + n_actions, 300),
            nn.ReLU()
        )

        self.final_layer = nn.Sequential(
            nn.Linear(300, 1)
        )

        self.first_layer.apply(self.init_weights)
        self.model.apply(self.init_weights)
        self.final_layer.apply(self.final_layer_init_weights)
    
    @torch.no_grad()
    def final_layer_init_weights(self, m):
        value = 3e-4
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight, -value, value)

    @torch.no_grad()
    def init_weights(self, m):
        if type(m) == nn.Linear:
            f = m.weight.size()[0]
            value = 1.0 / np.sqrt(f)
            nn.init.uniform_(m.weight, -value, value)        

    def forward(self, state, action):
        intermediate = self.first_layer(state)
        intermediate = torch.cat([intermediate, action], dim = 1)
        output = self.model(intermediate)
        output = self.final_layer(output)
        return output


class DDPG:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name) 
        self.device = torch.device('cuda')

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]  

        self.actor_network = Actor(self.n_states, self.n_actions).to(self.device)
        self.critic_network = Critic(self.n_states, self.n_actions).to(self.device)
        self.target_actor_network = Actor(self.n_states, self.n_actions).to(self.device)
        self.target_critic_network = Critic(self.n_states, self.n_actions).to(self.device)
        self.initialize_target_networks()

        self.actor_learning_rate = 1e-4
        self.critic_learning_rate = 1e-3
        self.gamma = 0.99
        self.tau = 0.001
        self.replay_buffer_size = int(1e6)
        self.batch_size = 64
        self.training_start = 25000
        self.critic_weight_decay = 1e-2
        self.total_episodes = 20000
        self.steps = 0

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr = self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr = self.critic_learning_rate, 
            weight_decay = self.critic_weight_decay)
        self.critic_loss_criterion = nn.MSELoss()
        self.ou_noise = OUNoise(self.n_actions)
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        self.total_rewards = []
        self.mean_reward = None
    
    def save_networks(self):
        torch.save(self.actor_network.state_dict(), self.env_name + '-myddpgactor.pth')
        torch.save(self.critic_network.state_dict(), self.env_name + '-myddpgcritic.pth')

    def load_networks(self):
        self.actor_network.load_state_dict(torch.load(self.env_name + '-myddpgactor.pth'))
        self.critic_network.load_state_dict(torch.load(self.env_name + '-myddpgcritic.pth'))
        self.initialize_target_networks()

    def initialize_target_networks(self):
        self.target_actor_network.load_state_dict(self.actor_network.state_dict())
        self.target_critic_network.load_state_dict(self.critic_network.state_dict())  

    def update_target_networks(self):
        for target_param, param in zip(self.target_actor_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        
        for target_param, param in zip(self.target_critic_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def train(self):
        state = self.env.reset()
        total_reward = 0.0
        self.ou_noise.reset()
        episode_num = 0

        self.writer = SummaryWriter(comment = '-myddpg' + self.env_name)
        while episode_num < self.total_episodes:
            self.steps += 1
            if self.steps <= self.training_start:
                action = self.env.action_space.sample()
            else:
                state_t = state.copy()
                state_t = torch.Tensor(state_t).to(self.device)
                action = self.actor_network(state_t).detach().cpu().numpy()
                noise = self.ou_noise.noise()
                action = action + noise
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            experience = Experience(state, action, next_state, reward, done)
            self.replay_buffer.append(experience)

            if self.steps > self.training_start:
                batch = self.replay_buffer.sample(self.batch_size)
                states, actions, next_states, rewards, dones = batch

                states = torch.tensor(states).float().to(self.device)
                actions = torch.tensor(actions).float().to(self.device)
                next_states = torch.tensor(next_states).float().to(self.device)
                terminals = torch.tensor(dones).to(self.device).view(-1, 1)
                rewards = torch.tensor(rewards).to(self.device).view(-1, 1)

                Q_vals = self.critic_network(states, actions)
                next_state_actions = self.target_actor_network(next_states)
                Q_targets = rewards + self.gamma * self.target_critic_network(next_states, next_state_actions) * (1 - terminals)
                critic_loss = self.critic_loss_criterion(Q_vals, Q_targets)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                actor_loss = -self.critic_network(states, self.actor_network(states)).mean()
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
                episode_num += 1
                total_reward = 0.0

        self.writer.close()       

    def run_test_episode(self, episode_num):
        self.env.render()
        state = self.env.reset()
        total_reward = 0
        while True:
            self.env.render()

            state_t = state.copy()
            state_t = torch.Tensor(state_t).to(self.device)
            action = self.actor_network(state_t).detach().cpu().numpy()

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
    ddpg = DDPG(env_name)
    ddpg.test()