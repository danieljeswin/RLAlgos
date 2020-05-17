import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd



class NoisyLinearFunction(autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, sigma_weight, sigma_bias, epsilon_input, epsilon_output):  
        output_features, input_features = weight.size()

        transform = lambda x : torch.sign(x) * torch.sqrt(torch.abs(x))
        epsilon_input = transform(epsilon_input)
        epsilon_output = transform(epsilon_output)

        epsilon_bias = epsilon_output

        bias = bias + sigma_bias * epsilon_output
        epsilon_output = epsilon_output.unsqueeze(1)
        epsilon_input = epsilon_input.unsqueeze(0)
        epsilon_weight = epsilon_output * epsilon_input

        ctx.save_for_backward(input, weight, sigma_weight, epsilon_weight, epsilon_output)        
        weight = weight + sigma_weight * epsilon_weight

        output = input.mm(weight.t())
        output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, sigma_weight, epsilon_weight, epsilon_output = ctx.saved_tensors

        grad_input = grad_output.mm(weight + sigma_weight * epsilon_weight)
        grad_weight = grad_output.t().mm(input)
        grad_bias = grad_output.sum(0)

        grad_sigma_weight = epsilon_weight * grad_weight
        grad_sigma_bias = epsilon_output * grad_bias

        return grad_input, grad_weight, grad_bias, grad_sigma_weight, grad_sigma_bias, None, None


class NoisyLinearLayer(nn.Module):
    def __init__(self, input_features, output_features, sigma = 0.017):
        super(NoisyLinearLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features


        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias = nn.Parameter(torch.Tensor(output_features))
        self.sigma_weight = nn.Parameter(torch.full((output_features, input_features), sigma))
        self.sigma_bias = nn.Parameter(torch.full((output_features,), sigma))

        self.register_buffer("epsilon_input", torch.zeros(input_features))
        self.register_buffer("epsilon_output", torch.zeros(output_features))

        std = np.sqrt(3 / input_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)
    
    def forward(self, input):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()
        return NoisyLinearFunction.apply(input, self.weight, self.bias, self.sigma_weight, self.sigma_bias, self.epsilon_input, self.epsilon_output)



class DQNModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = input_shape[0], out_channels = 32, kernel_size = 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
            nn.ReLU()
        )

        conv_out_size = self.get_output_size(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def get_output_size(self, input_shape):
        temp = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(temp.size()))

    def forward(self, state):
        conv_out = self.conv(state).view(state.size()[0], -1)
        return self.fc(conv_out)

class DQNNoisyModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNNoisyModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = input_shape[0], out_channels = 32, kernel_size = 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
            nn.ReLU()
        )
        conv_out_size = self.get_output_size(input_shape)
        self.fc = nn.Sequential(
            NoisyLinearLayer(input_features = conv_out_size, output_features = 512),
            nn.ReLU(),
            NoisyLinearLayer(input_features = 512, output_features = n_actions)
        )

    def get_output_size(self, input_shape):
        temp = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(temp.size()))
    
    def forward(self, state):
        conv_out = self.conv(state).view(state.size()[0], -1)
        return self.fc(conv_out)
    

class DQNLinearModel(nn.Module):
    def __init__(self, states_length, n_actions):
        super(DQNLinearModel, self).__init__()


        self.fc = nn.Sequential(
            nn.Linear(states_length, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, state):
        return self.fc(state)

class DuelingDQNModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQNModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = input_shape[0], out_channels = 32, kernel_size = 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
            nn.ReLU()
        )
        conv_out_size = self.get_output_size(input_shape)

        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)    
        )
    
    def get_output_size(self, input_shape):
        temp = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(temp.size()))

    def forward(self, state):
        conv_out = self.conv(state).view(state.size()[0], -1)
        value = self.fc_value(conv_out)
        advantage = self.fc_advantage(conv_out)

        return value + advantage - advantage.mean()
