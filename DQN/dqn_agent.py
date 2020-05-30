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
        self.env = wrappers.make_env(env_name)


        self.env_name = env_name
        self.device = torch.device('cuda')

        self.learning_rate = 2.5e-4
        self.stabilizer = 0.01
        self.gradient_momentum = 0.95
        self.gamma = 0.99
        self.batch_size = 32
        self.replay_start_size = 50000
        self.replay_buffer_size = 1000000
        self.update_target_interval = 10000
        self.training_frequency = 4

        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_period = 1000000

        self.network = dqn_model.DQNModel(self.env.observation_space.shape, self.env.action_space.n).to(self.device)
        self.target_network = dqn_model.DQNModel(self.env.observation_space.shape, self.env.action_space.n).to(self.device)
        self.replay_buffer = replay_buffer.ReplayBuffer(self.replay_buffer_size)
        self.optimizer = optim.RMSprop(self.network.parameters(), lr = self.learning_rate, alpha = self.gradient_momentum, 
                    eps = self.stabilizer)
        self.loss_criterion = nn.SmoothL1Loss()

        print(self.network)

        self.writer = SummaryWriter(comment = 'dqn' + self.env_name)
        self.total_rewards = []
        self.best_mean_reward = None

        self.steps = 0

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())
    
    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            state_t = np.array([state], copy = False)
            state_t = torch.tensor(state_t).float().to(self.device)
            q_vals = self.network(state_t)

            _, action = torch.max(q_vals, dim = 1)
            action = int(action.item())
        return action
    
    def optimize(self):
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, next_states, rewards, terminals = batch

        states = torch.tensor(states).float().to(self.device)
        next_states = torch.tensor(next_states).float().to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        terminals = torch.BoolTensor(terminals).to(self.device)

        Q_vals = self.network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        Q_targets = self.target_network(next_states).max(1)[0]
        Q_targets[terminals] = 0.0
        Q_targets = Q_targets.detach()

        Q_targets = Q_targets * self.gamma + rewards
        
        self.optimizer.zero_grad()
        loss = self.loss_criterion(Q_vals, Q_targets)
        loss.backward()
        self.optimizer.step()

    def get_epsilon(self):
        if self.steps <= self.replay_start_size:
            return self.epsilon_start
        steps = self.steps - self.replay_start_size
        epsilon = self.epsilon_start - steps * (self.epsilon_start - self.epsilon_end) / self.epsilon_period

        return max(self.epsilon_end, epsilon)

    def run_train_episode(self):
        state = self.env.reset()
        total_reward = 0.0
        frame_index = self.steps
        start_time = time.time()

        while True:
            self.steps += 1
            epsilon = self.get_epsilon()

            action = self.get_action(state, epsilon)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            experience = replay_buffer.Experience(state, action, next_state, reward, done)
            self.replay_buffer.append(experience)

            if self.steps % self.update_target_interval == 0:
                self.update_target_network()
            
            if len(self.replay_buffer) >= self.replay_start_size and self.steps % self.training_frequency == 0:
                self.optimize()
            
            state = next_state
            if done:
                if self.env.was_real_done:

                    self.total_rewards.append(total_reward)
                    speed = (self.steps - frame_index) / (time.time() - start_time)
                    mean_reward = np.mean([self.total_rewards[-100:]])

                    print("%d: Done %d games, mean reward %.3f, epsilon %.2f, speed %.2f f/s" % (self.steps, len(self.total_rewards), mean_reward, 
                        epsilon, speed))

                    self.writer.add_scalar("epsilon", epsilon, self.steps)
                    self.writer.add_scalar("speed", speed, self.steps)
                    self.writer.add_scalar("mean_reward", mean_reward, self.steps)
                    self.writer.add_scalar("reward", total_reward, self.steps)

                    if self.best_mean_reward is None or self.best_mean_reward < mean_reward:
                        torch.save(self.network.state_dict(), self.env_name + "-mybest.pth")    
                        if self.best_mean_reward is not None:
                            print("Best mean reward updated %.3f -> %.3f, model saved" % (self.best_mean_reward, mean_reward))
                        self.best_mean_reward = mean_reward

                    break
                else:
                    state = self.env.reset()
                


    def train(self):
        for _ in range(100000):
            self.run_train_episode()
        self.writer.close()
    
    def run_test_episode(self):
        self.env = gym.wrappers.Monitor(self.env, directory = self.env_name, force = True)
        self.network.load_state_dict(torch.load(self.env_name + "-mybest.pth"))

        state = self.env.reset()
        total_reward = 0.0
        c = collections.Counter()
        while True:
            start_ts = time.time()
            self.env.render()
            state_t = torch.tensor(np.array([state], copy = False)).float().to(self.device)
            q_vals = self.network(state_t).detach().cpu().numpy()[0]
            action = np.argmax(q_vals)
            c[action] += 1

            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            state = next_state
            if done:
                if self.env.was_real_done:
                    break
                else:
                    state = self.env.reset()
            
            delta = 1/25 - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
        
        print("Total reward : %.2f" % total_reward)
        print("Action counts : ", c)
    
    def test(self):
        for _ in range(20):
            self.run_test_episode()

if __name__ == "__main__":
    env_name = 'BreakoutNoFrameskip-v4'
    agent = DQNAgent(env_name)
    agent.test()