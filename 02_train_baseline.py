from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

if __name__ == "__main__":
    env = make_vec_env("LunarLander-v3", n_envs=16)

    model = PPO(
        "MlpPolicy", env,
        n_steps=1024, batch_size=64, n_epochs=4,
        gamma=0.999, gae_lambda=0.98, ent_coef=0.01,
        verbose=1, seed=0,
    )
    model.learn(total_timesteps=1_000_000)
    model.save("models/ppo_baseline")