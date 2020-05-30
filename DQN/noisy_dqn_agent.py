from lib import wrappers
from lib import replay_buffer
from lib import dqn_model

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gym
from torch.utils.tensorboard import SummaryWriter
import argparse
import time

import collections

class NoisyDQN:
    def __init__(self, env_name):
        self.env = wrappers.make_env(env_name)
        self.device = torch.device('cuda')
        self.env_name = env_name

        torch.manual_seed(2)
        np.random.seed(2)
        self.env.seed(2)
        self.env.action_space.seed(2)
        self.env.observation_space.seed(2)

        self.replay_buffer_size = 10000
        self.replay_start_size = 10000
        self.update_interval = 1000
        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.batch_size = 32

        n_actions = self.env.action_space.n
        input_shape = self.env.observation_space.shape
        self.network = dqn_model.DQNNoisyModel(input_shape, n_actions).to(self.device)
        self.target_network = dqn_model.DQNNoisyModel(input_shape, n_actions).to(self.device)
        self.replay_buffer = replay_buffer.ReplayBuffer(self.replay_buffer_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr = self.learning_rate)

        print(self.network)

        self.writer = SummaryWriter(comment = 'noisy_dqn' + env_name)
        self.total_rewards = []
        self.best_mean_reward = None

        self.steps = 0

    def get_action(self, state):
        state_t = np.array([state], copy = False)
        state_t = torch.tensor(state_t).to(self.device)
        q_vals = self.network(state_t)

        _, action = torch.max(q_vals, dim = 1)
        action = int(action.item())
        return action

    def update_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def save_network(self):
        torch.save(self.network.state_dict(), self.env_name + "-mynoisydqnbest.pth")
    
    def load_network(self):
        self.network.load_state_dict(torch.load(self.env_name + "-mynoisydqnbest.pth"))
    
    def optimize(self):
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, next_states, rewards, terminals = batch

        states = torch.tensor(states).to(self.device)
        next_states = torch.tensor(next_states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        terminals = torch.BoolTensor(terminals).to(self.device)

        Q_vals = self.network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        Q_targets = self.target_network(next_states).max(1)[0]
        Q_targets = Q_targets.detach()
        Q_targets[terminals] = 0.0
        Q_targets = rewards + self.gamma * Q_targets

        loss = nn.MSELoss()(Q_vals, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run_train_episode(self):
        state = self.env.reset()
        total_reward = 0.0
        frame_index = self.steps
        start_time = time.time()

        while True:
            self.steps += 1

            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            experience = replay_buffer.Experience(state, action, next_state, reward, done)
            self.replay_buffer.append(experience)

            if self.steps % self.update_interval == 0:
                self.update_network()
            
            if len(self.replay_buffer) >= self.replay_start_size:
                self.optimize()

            if done:
                self.total_rewards.append(total_reward)
                end_time = time.time()
                speed = (self.steps - frame_index) / (end_time - start_time)
                mean_rewards = np.mean([self.total_rewards[-100:]])

                self.writer.add_scalar("speed", speed, self.steps)
                self.writer.add_scalar("mean_reward", mean_rewards, self.steps)
                self.writer.add_scalar("reward", total_reward, self.steps)

                print("%d: Done %d games, mean reward %.3f, speed %.2f f/s" % (self.steps, len(self.total_rewards), mean_rewards, 
                        speed))

                if self.best_mean_reward is None or self.best_mean_reward < mean_rewards:
                    if self.best_mean_reward is not None:
                        print("Best mean reward updated %.3f -> %.3f, model saved" % (self.best_mean_reward, mean_rewards))
                    self.best_mean_reward = mean_rewards
                    self.save_network()
                break
            state = next_state

    def run_test_episode(self):

        total_reward = 0.0
        self.load_network()
        self.env = gym.wrappers.Monitor(self.env, directory = self.env_name, force = True)
        state = self.env.reset()
        c = collections.Counter()
        while True:
            start_ts = time.time()
            self.env.render()

            state_t = np.array([state], copy = False)
            state_t = torch.tensor(state_t).to(self.device)
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


    def train(self):
        for _ in range(400):
            self.run_train_episode()
        self.writer.close()

    def test(self):
        for _ in range(20):
            self.run_test_episode()
        self.writer.close()

if __name__ == '__main__':
    env_name = 'PongNoFrameskip-v4'
    dqn = NoisyDQN(env_name)
    dqn.train()
