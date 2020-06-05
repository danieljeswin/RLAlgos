import torch
import torch.nn as nn
import torch.optim as optim

import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
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

class ValueCritic(nn.Module):
    def __init__(self, n_states):
        super(ValueCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        output = self.fc(x)
        return output

class A2C:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)

        self.device = torch.device('cuda')
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.gamma = 0.99
        self.learning_rate = 7e-4
        self.total_episodes = 3000
        self.trajectory_length = 500
        self.steps = 0
        self.entropy_coefficient = 0.01
        self.critic_coefficient = 0.5
        self.max_grad_norm = 0.5
        
        self.policy_network = PolicyNetwork(self.n_states, self.n_actions).to(self.device)
        self.value_critic = ValueCritic(self.n_states).to(self.device)
        self.policy_network_optimizer = optim.Adam(self.policy_network.parameters(), lr = self.learning_rate)
        self.value_critic_optimizer = optim.Adam(self.value_critic.parameters(), lr = self.learning_rate)

        self.critic_loss_criterion = nn.MSELoss()

        print(self.policy_network)
        print(self.value_critic)

        self.total_rewards = []
        self.mean_reward = None
    
    def save_network(self):
        torch.save(self.policy_network.state_dict(), self.env_name + "-mya2cactor.pth")
        torch.save(self.value_critic.state_dict(), self.env_name + "-mya2ccritic.pth")


    def load_network(self):
        self.policy_network.load_state_dict(torch.load(self.env_name + "-mya2cactor.pth"))
        self.value_critic.load_state_dict(torch.load(self.env_name + "-mya2ccritic.pth"))

    def get_discounted_rewards(self, rewards, dones):
        for t in range(self.trajectory_length - 2, -1, -1):
            rewards[t] += self.gamma * rewards[t + 1] * (1 - dones[t])
        return rewards

    def train(self):
        self.writer = SummaryWriter(comment = '-mya2c')
        episode_num = 0
        state = self.env.reset()
        total_reward = 0.0
        while episode_num < self.total_episodes:  

            # Initialize values
            rewards = np.zeros(self.trajectory_length)
            logprob = torch.zeros((self.trajectory_length)).float().to(self.device)
            entropy = torch.zeros((self.trajectory_length)).float().to(self.device)
            values = torch.zeros((self.trajectory_length)).float().to(self.device)
            dones = np.zeros(self.trajectory_length)
            step = 0
            for step in range(self.trajectory_length):
                self.steps += 1

                # Select action and get logprob, entropy
                state_t = state.copy()
                state_t = torch.Tensor(state_t).to(self.device)
                logits = self.policy_network(state_t)
                values[step] = self.value_critic(state_t)
                distribution = Categorical(logits = logits)
                action = distribution.sample()
                logprob[step] = distribution.log_prob(action)
                entropy[step] = distribution.entropy()
                action = action.item()

                next_state, rewards[step], dones[step], _ = self.env.step(action)
                total_reward += rewards[step]

                state = next_state

                if dones[step]:
                    episode_num += 1
                    state = self.env.reset()
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
                    total_reward = 0.0
                

            if not dones[step]:
                state_t = state.copy()
                state_t = torch.Tensor(state_t).to(self.device)
                value = self.value_critic(state_t).item()
                rewards[step] += self.gamma * value
            
            discounted_rewards = self.get_discounted_rewards(rewards, dones)
            advantages = discounted_rewards - values.detach().cpu().numpy()
            advantages = torch.Tensor(advantages).to(self.device)
            actor_loss = -logprob * advantages

            discounted_rewards = torch.Tensor(discounted_rewards).to(self.device)
            critic_loss = self.critic_loss_criterion(discounted_rewards, values) * self.critic_coefficient
            self.value_critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.value_critic.parameters(), self.max_grad_norm)
            self.value_critic_optimizer.step()

            policy_loss = (actor_loss - entropy * self.entropy_coefficient).mean()
            self.policy_network_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
            self.policy_network_optimizer.step()

        self.writer.close()

    def run_test_episode(self, episode_num):
        state = self.env.reset()
        total_reward = 0
        while True:
            self.env.render()

            state_t = state.copy()
            state_t = torch.Tensor(state_t).to(self.device)
            logits = self.policy_network(state_t)
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
    env_name = 'CartPole-v0'
    a2c = A2C(env_name)
    a2c.train()