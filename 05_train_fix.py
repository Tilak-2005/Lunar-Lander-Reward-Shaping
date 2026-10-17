import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from wrappers import ImprovementBonusWrapper

if __name__ == "__main__":
    torch.set_num_threads(1)

    env = make_vec_env(
        "LunarLander-v3", n_envs=8,
        wrapper_class=ImprovementBonusWrapper,
        vec_env_cls=SubprocVecEnv,
    )

    model = PPO(
        "MlpPolicy", env,
        n_steps=1024, batch_size=64, n_epochs=4,
        gamma=0.999, gae_lambda=0.98, ent_coef=0.01,
        device="cpu", verbose=1, seed=0,
    )
    try:
        model.learn(
            total_timesteps=1_000_000,
            callback=CheckpointCallback(50_000, "models/checkpoints", name_prefix="fix"),
        )
    finally:
        model.save("models/ppo_fixed")