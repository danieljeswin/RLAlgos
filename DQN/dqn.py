from lib import wrappers
from lib import dqn_model
from lib import replay_buffer

import torch
import torch.nn as nn
import torch.optim as optim
import gym

from torch.utils.tensorboard import SummaryWriter
import argparse
import time

import numpy as np
import collections

class DQNAgent:
    def __init__(self, env_name):
        self.device = torch.device('cuda')
        self.env_name = env_name
        self.env = wrappers.make_env(env_name)

        self.gamma = 0.99
        self.batch_size = 32
        self.replay_buffer_size = 10000
        self.replay_start_size = 10000
        self.learning_rate = 1e-4
        self.update_target_interval = 1000

        self.epsilon_start = 1.0
        self.epsilon_end = 0.02
        self.epsilon_period = 100000

        self.reward_bound = 19.5

        self.replay_buffer = replay_buffer.ReplayBuffer(self.replay_buffer_size)
        self.network = dqn_model.DQNModel(self.env.observation_space.shape, self.env.action_space.n).to(self.device)
        self.target_network = dqn_model.DQNModel(self.env.observation_space.shape, self.env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr = self.learning_rate)

        print(self.network)

        self.writer = SummaryWriter(comment = 'dqn' + self.env_name)

        self.total_rewards = []
        self.frame_index = 0

    def play_step(self, epsilon, state, total_reward):

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_t = np.array([state], copy = False)
            state_t = torch.tensor(state_t).to(self.device)
            q_vals = self.network(state_t)
            _, action = torch.max(q_vals, dim = 1)
            action = int(action.item())
        
        next_state, reward, done, _ = self.env.step(action)
        total_reward += reward

        experience = replay_buffer.Experience(state, action, next_state, reward, done)
        self.replay_buffer.append(experience)
        state = next_state

        return state, total_reward, done
    
    def calc_loss(self):
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, next_states, rewards, terminals = batch

        states = torch.tensor(states).to(self.device)
        next_states = torch.tensor(next_states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        terminals = torch.BoolTensor(terminals).to(self.device)

        Q_vals = self.network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        Q_targets = self.target_network(next_states).max(1)[0]
        Q_targets[terminals] = 0.0
        Q_targets = Q_targets.detach()

        Q_targets = Q_targets * self.gamma + rewards

        return nn.MSELoss()(Q_vals, Q_targets)

    def train(self):
        ts_frame = 0
        ts = time.time()
        best_mean_reward = None
        epsilon = self.epsilon_start

        state = self.env.reset()
        total_reward = 0.0

        while True:
            self.frame_index += 1
            epsilon = max(self.epsilon_end, self.epsilon_start - self.frame_index / self.epsilon_period)

            state, total_reward, done = self.play_step(epsilon, state, total_reward)
            if done:
                self.total_rewards.append(total_reward)
                speed = (self.frame_index - ts_frame) / (time.time() - ts)
                ts_frame = self.frame_index
                ts = time.time()

                mean_reward = np.mean(self.total_rewards[-100:])
                print("%d: Done %d games, mean reward %.3f, epsilon %.2f, speed %.2f f/s" % (self.frame_index, len(self.total_rewards), mean_reward, 
                            epsilon, speed))

                self.writer.add_scalar("epsilon", epsilon, self.frame_index)
                self.writer.add_scalar("speed", speed, self.frame_index)
                self.writer.add_scalar("mean_reward", mean_reward, self.frame_index)
                self.writer.add_scalar("reward", total_reward, self.frame_index)

                total_reward = 0.0
                state = self.env.reset()

                if best_mean_reward is None or best_mean_reward < mean_reward:
                    torch.save(self.network.state_dict(), self.env_name + "-best1.pth")    
                    if best_mean_reward is not None:
                        print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                    best_mean_reward = mean_reward

                if mean_reward > self.reward_bound:
                    print("Solved in %d frames!" % self.frame_index)
                    break
            
            if len(self.replay_buffer) < self.replay_start_size:
                continue
            
            if self.frame_index % self.update_target_interval == 0:
                self.target_network.load_state_dict(self.network.state_dict())

            self.optimizer.zero_grad()
            loss = self.calc_loss()
            loss.backward()
            self.optimizer.step()
        
        self.writer.close()

    def test(self):
        self.env = gym.wrappers.Monitor(self.env, directory = self.env_name, force = True)
        self.network.load_state_dict(torch.load(self.env_name + "-best.pth"))

        state = self.env.reset()
        total_reward = 0.0
        c = collections.Counter()
        while True:
            start_ts = time.time()
            self.env.render()
            state_t = torch.tensor(np.array([state], copy = False)).to(self.device)
            q_vals = self.network(state_t).detach().cpu().numpy()[0]
            action = np.argmax(q_vals)
            c[action] += 1

            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            if done:
                break
            state = next_state
            delta = 1/25 - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
        
        print("Total reward : %.2f" % total_reward)
        print("Action counts : ", c)
    
    def test_epsiodes(self):
        for _ in range(20):
            self.test()

if __name__ == "__main__":
    env_name = 'PongNoFrameskip-v4'
    agent = DQNAgent(env_name)
    agent.test_epsiodes()
