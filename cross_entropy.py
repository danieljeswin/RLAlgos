import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
from observation_wrapper import OneHotWrapper

from cross_entropy_net import CrossEntropyNet
import random

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


Episode = namedtuple('Episode', field_names = ['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names = ['observation', 'action'])

def iterate_batches(env, net, batch_size = 16):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    state = env.reset()

    softmax = nn.Softmax(dim = 1)
    while True:
        state_v = torch.cuda.FloatTensor([state])
        action_probs = softmax(net(state_v))
        action_probs = action_probs.cpu().detach().numpy()[0]

        action = np.random.choice(len(action_probs), p = action_probs)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward

        episode_steps.append(EpisodeStep(state, action))
        if done:
            batch.append(Episode(episode_reward, episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_state = env.reset()

            if len(batch) == batch_size:
                yield batch
                batch = []
        state = next_state

# def filter_batch(batch, percentile):
#     rewards = list(map(lambda s: s.reward, batch))
#     rewards_bound = np.percentile(rewards, percentile)
#     rewards_mean = float(np.mean(rewards))

#     train_states = []
#     train_actions = []
#     for example in batch:
#         if example.reward < rewards_bound:
#             continue
        
#         train_states.extend(map(lambda step: step.observation, example.steps))
#         train_actions.extend(map(lambda step: step.action, example.steps))

#     train_states = torch.cuda.FloatTensor(train_states)
#     train_actions = torch.cuda.LongTensor(train_actions)
#     return train_states, train_actions, rewards_bound, rewards_mean

def filter_batch(batch, percentile):
    disc_rewards = list(map(lambda s: s.reward * (0.9 ** len(s.steps)), batch))
    reward_bound = np.percentile(disc_rewards, percentile)

    train_states = []
    train_actions = []
    elite_batch = []
    for example, discounted_reward in zip(batch, disc_rewards):
        if discounted_reward > reward_bound:
            train_states.extend(map(lambda step: step.observation, example.steps))
            train_actions.extend(map(lambda step: step.action, example.steps))
            elite_batch.append(example)

    train_states = torch.cuda.FloatTensor(train_states)
    train_actions = torch.cuda.LongTensor(train_actions)
    return elite_batch, train_states, train_actions, reward_bound
        



if __name__ == "__main__":
    random.seed(12345)
    device = torch.device('cuda')
    # env = gym.make('CartPole-v0')
    env = OneHotWrapper(gym.make('FrozenLake-v0'))
    # env = gym.wrappers.Monitor(env, "cross_entropy", force = True)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    network = CrossEntropyNet(n_states, HIDDEN_SIZE, n_actions).to(device)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = network.parameters(), lr = 0.001)
    writer = SummaryWriter()

    full_batch = []
    for iter_num, batch in enumerate(iterate_batches(env, network, BATCH_SIZE)):
        reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
        # states, actions, reward_bound, reward_mean = filter_batch(batch, PERCENTILE)
        full_batch, states, actions, reward_bound = filter_batch(batch, PERCENTILE)
        if not full_batch:
            continue
        full_batch = full_batch[-500:]

        optimizer.zero_grad()
        action_preds = network(states)
        loss = loss_criterion(action_preds, actions)
        loss.backward()
        optimizer.step()

        print("%d : loss = %.3f, reward_mean = %.1f, reward_bound = %.1f" % (iter_num, loss.item(), reward_mean, reward_bound))
        writer.add_scalar("loss", loss.item(), iter_num)
        writer.add_scalar("Mean Reward", reward_mean, iter_num)
        writer.add_scalar("Boundary Reward", reward_bound, iter_num)

        if reward_mean > 0.8:
            print("Environment Solved")
            break
    writer.close()
