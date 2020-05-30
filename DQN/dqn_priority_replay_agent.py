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

class DQNPriorityReplayAgent:
    def __init__(self, env_name):
        self.env = wrappers.make_env(env_name)
        self.env_name = env_name
        self.device = torch.device('cuda')

        torch.manual_seed(2)
        np.random.seed(2)
        self.env.seed(2)
        self.env.action_space.seed(2)
        self.env.observation_space.seed(2)


        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.batch_size = 32
        self.replay_start_size = 10000
        self.replay_buffer_size = 10000
        self.update_target_interval = 1000

        self.epsilon_start = 1.0
        self.epsilon_end = 0.02
        self.epsilon_period = 100000

        self.alpha = 0.6
        self.beta_start = 0.4
        self.beta_period = 100000
        self.beta = self.beta_start

        self.network = dqn_model.DQNModel(self.env.observation_space.shape, self.env.action_space.n).to(self.device)
        self.target_network = dqn_model.DQNModel(self.env.observation_space.shape, self.env.action_space.n).to(self.device)
        self.replay_buffer = replay_buffer.PriorityBuffer(self.replay_buffer_size, self.alpha)
        self.optimizer = optim.Adam(self.network.parameters(), lr = self.learning_rate)

        print(self.network)

        self.writer = SummaryWriter(comment = 'dqnpriority' + self.env_name)
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
            state_t = torch.tensor(state_t).to(self.device)
            q_vals = self.network(state_t)

            _, action = torch.max(q_vals, dim = 1)
            action = int(action.item())
        return action
    
    def optimize(self):
        batch = self.replay_buffer.sample(self.batch_size, self.beta)
        states, actions, next_states, rewards, terminals, indices, weights = batch

        states = torch.tensor(states).to(self.device)
        next_states = torch.tensor(next_states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        terminals = torch.BoolTensor(terminals).to(self.device)
        weights = torch.tensor(weights).to(self.device)

        Q_vals = self.network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        Q_targets = self.target_network(next_states).max(1)[0]
        Q_targets[terminals] = 0.0
        Q_targets = Q_targets.detach()

        Q_targets = Q_targets * self.gamma + rewards
        
        self.optimizer.zero_grad()
        loss_t = weights * (Q_vals - Q_targets) ** 2
        loss = loss_t.mean()
        loss.backward()
        self.optimizer.step()

        priorities = 1e-5 + torch.abs(Q_vals - Q_targets)
        self.replay_buffer.update_priorities(indices, priorities)

    def run_train_episode(self):
        state = self.env.reset()
        total_reward = 0.0
        frame_index = self.steps
        start_time = time.time()

        while True:
            self.steps += 1
            epsilon = max(self.epsilon_end, self.epsilon_start - self.steps * (self.epsilon_start - self.epsilon_end) / self.epsilon_period)
            self.beta = min(1.0, self.beta_start + self.steps * (1 - self.beta_start) / self.beta_period)

            action = self.get_action(state, epsilon)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            experience = replay_buffer.Experience(state, action, next_state, reward, done)
            self.replay_buffer.append(experience)

            if self.steps % self.update_target_interval == 0:
                self.update_target_network()
            
            if len(self.replay_buffer) >= self.replay_start_size:
                self.optimize()
            
            if done:
                self.total_rewards.append(total_reward)
                speed = (self.steps - frame_index) / (time.time() - start_time)
                mean_reward = np.mean([self.total_rewards[-100:]])

                print("%d: Done %d games, mean reward %.3f, epsilon %.2f, beta %.2f, speed %.2f f/s" % (self.steps, len(self.total_rewards), mean_reward, 
                    epsilon, self.beta, speed))

                self.writer.add_scalar("epsilon", epsilon, self.steps)
                self.writer.add_scalar("speed", speed, self.steps)
                self.writer.add_scalar("mean_reward", mean_reward, self.steps)
                self.writer.add_scalar("reward", total_reward, self.steps)
                self.writer.add_scalar("beta", self.beta, self.steps)

                if self.best_mean_reward is None or self.best_mean_reward < mean_reward:
                    torch.save(self.network.state_dict(), self.env_name + "-mydqnprioritybest.pth")    
                    if self.best_mean_reward is not None:
                        print("Best mean reward updated %.3f -> %.3f, model saved" % (self.best_mean_reward, mean_reward))
                    self.best_mean_reward = mean_reward

                break
            state = next_state


    def train(self):
        for _ in range(400):
            self.run_train_episode()
        self.writer.close()
    
    def run_test_episode(self):
        self.env = gym.wrappers.Monitor(self.env, directory = self.env_name, force = True)
        self.network.load_state_dict(torch.load(self.env_name + "-mydqnprioritybest.pth"))

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
    
    def test(self):
        for _ in range(20):
            self.run_test_episode()

if __name__ == "__main__":
    env_name = 'PongNoFrameskip-v4'
    agent = DQNPriorityReplayAgent(env_name)
    agent.train()