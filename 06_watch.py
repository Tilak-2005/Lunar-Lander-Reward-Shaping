import sys
import gymnasium as gym
from stable_baselines3 import PPO

name = sys.argv[1]
model = PPO.load(f"models/{name}")

env = gym.make("LunarLander-v3", render_mode="human")
for _ in range(5):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        done = term or trunc
env.close()