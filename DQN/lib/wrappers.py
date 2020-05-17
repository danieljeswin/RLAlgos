import cv2
import gym
import gym.spaces
import numpy as np
import collections

class FireResetWrapper(gym.Wrapper):
    def __init__(self, env = None):
        super(FireResetWrapper, self).__init__(env)
        
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)
    
    def reset(self):
        self.env.reset()
        next_state, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        
        next_state, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()

        return next_state
    
class MaxAndSkipWrapper(gym.Wrapper):
    def __init__(self, env = None, skip = 4):
        super(MaxAndSkipWrapper, self).__init__(env)
        self.obs_buffer = collections.deque(maxlen = 2)
        self.skip = 4

    def step(self, action):
        total_reward = 0.0

        for _ in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            self.obs_buffer.append(state)

            if done:
                break
        max_frame = np.max(np.stack(self.obs_buffer), axis = 0)
        return max_frame, total_reward, done, info

    def reset(self):
        self.obs_buffer.clear()
        state = self.env.reset()
        self.obs_buffer.append(state)

        return state

class ResizeFrameWrapper(gym.ObservationWrapper):
    def __init__(self, env = None):
        super(ResizeFrameWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(low = 0, high = 255, shape = (84, 84, 1), dtype = np.uint8)
    
    def observation(self, state):
        return ResizeFrameWrapper.process(state)
    
    @staticmethod
    def process(state):
        if state.size == 210 * 160 * 3:
            image = np.reshape(state, [210, 160, 3]).astype(np.float32)
        elif state.size == 250 * 160 * 3:
            image = np.reshape(state, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution"
        
        image = image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114
        resized_image = cv2.resize(image, (84, 110), interpolation = cv2.INTER_AREA)
        image = resized_image[18:102, :]
        image = np.reshape(image, [84, 84, 1])

        return image.astype(np.uint8)

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, steps, dtype = np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        self.steps = steps

        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(steps, axis = 0), old_space.high.repeat(steps, axis = 0), dtype = dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype = self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

class PytorchWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(PytorchWrapper, self).__init__(env)
        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low = 0.0, high = 0.0, shape = (shape[-1], shape[0], shape[1]), dtype = np.float32)
    
    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class ScaledFloatWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipWrapper(env)
    env = FireResetWrapper(env)
    env = ResizeFrameWrapper(env)
    env = PytorchWrapper(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatWrapper(env)

def make_breakout_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipWrapper(env)
    env = FireResetWrapper(env)
    env = ResizeFrameWrapper(env)
    env = PytorchWrapper(env)
    env = BufferWrapper(env, 4)
    return env


