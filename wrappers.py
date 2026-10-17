import math
import gymnasium as gym


class TimePenaltyWrapper(gym.Wrapper):
    """Small fee every frame, so waiting costs something."""
    def __init__(self, env, fee=0.05):
        super().__init__(env)
        self.fee = fee

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return obs, reward - self.fee, term, trunc, info


def is_upright_and_slow(obs):
    # obs: 0=x 1=y 2=vx 3=vy 4=angle 5=ang_vel
    return abs(obs[4]) < 0.2 and abs(obs[2]) < 0.2 and abs(obs[3]) < 0.2


class HoverBonusWrapper(gym.Wrapper):
    """BROKEN ON PURPOSE: +1 for every frame upright and slow."""
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        if is_upright_and_slow(obs):
            reward += 1.0
        return obs, reward, term, trunc, info


def potential(obs):
    # Higher = more upright and slower
    return -(abs(obs[4]) + math.hypot(obs[2], obs[3]))


class ImprovementBonusWrapper(gym.Wrapper):
    """FIX: pay for *improvement* (change in potential), plus a per-frame fee."""
    def __init__(self, env, scale=1.0, fee=0.05, gamma=0.999):
        super().__init__(env)
        self.scale, self.fee, self.gamma = scale, fee, gamma
        self._prev = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev = potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        phi = potential(obs)
        bonus = self.scale * (self.gamma * phi - self._prev)
        self._prev = phi
        return obs, reward + bonus - self.fee, term, trunc, info