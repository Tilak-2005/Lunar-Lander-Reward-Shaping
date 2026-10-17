import sys
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

name = sys.argv[1]   # e.g. ppo_baseline

model = PPO.load(f"models/{name}")

# 1) Score over 20 episodes, no wrapper
env = gym.make("LunarLander-v3")
mean, std = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
print(f"{name}: {mean:.1f} +/- {std:.1f}  (solved if mean >= 200)")
env.close()

# 2) Record 3 episodes
env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(env, video_folder=f"videos/{name}",
                  episode_trigger=lambda e: True, name_prefix=name)
for _ in range(3):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        done = term or trunc
env.close()