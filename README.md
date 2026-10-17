# Lunar Lander: Reward Shaping and Reward Hacking with PPO

Training a reinforcement learning agent to land a rocket in Gymnasium's `LunarLander-v3`, then showing how a badly designed reward makes the agent **hover forever instead of landing** (reward hacking), and how to fix it with improvement-based reward shaping.

| Baseline (lands) | Reward hack (hovers) | Fixed reward (lands) |
|---|---|---|
| `videos/ppo_baseline/` | `videos/ppo_hover_hack/` | `videos/ppo_fixed/` |

## Results

Evaluated on the **plain environment (no custom wrapper)** over 20 episodes, deterministic policy. A mean of 200 or more counts as solved.

| Model | Reward rule | Mean score | Behavior |
|---|---|---|---|
| `ppo_baseline` | Built-in reward | `<fill in>` +/- `<fill in>` | Lands |
| `ppo_hover_hack` | Built-in + 1 per frame while upright and slow | -97.4 +/- 36.1 | Does not land |
| `ppo_fixed` | Built-in + improvement bonus - per-frame fee | `<fill in>` +/- `<fill in>` | Lands |

> Replace the `<fill in>` values with the numbers printed by `python 03_evaluate.py <model_name>`.

## What this project shows

**Reinforcement learning (RL)** trains an agent by trial and error. The agent acts, receives a reward, and adjusts its behavior to earn more reward over time.

- **Environment:** `LunarLander-v3` (Gymnasium, Box2D physics). It runs at 50 frames per second.
- **Observation:** 8 numbers per frame: x position, y position, x velocity, y velocity, angle, angular velocity, and two leg-contact flags.
- **Actions:** 4 discrete choices: do nothing, fire left engine, fire main engine, fire right engine.
- **Algorithm:** PPO (Proximal Policy Optimization) from Stable-Baselines3 with an `MlpPolicy` (two hidden layers of 64 units). PPO keeps each policy update small so that one lucky run cannot destroy what the agent has learned.

### The reward hack

The hack wrapper adds **+1 for every frame the rocket is upright and slow**. An episode is cut off at 1,000 frames, so a rocket that hovers can collect up to about 1,000 bonus points while paying at most about 300 in fuel. Landing ends the episode and stops the payments, so hovering beats landing, and PPO learns exactly that. The training reward looks great while the real task score collapses.

### The fix

`ImprovementBonusWrapper` replaces the "sitting in a good spot" bonus with **potential-based shaping**: it pays for the *change* in a potential function (more upright and slower is better) between frames, and adds a small fee every frame.

```
bonus = scale * (gamma * potential(next_obs) - potential(obs)) - fee
```

Sitting still earns roughly nothing and every extra frame costs a little, so the best strategy is to improve and then finish the landing. Potential-based shaping is also known not to change which behavior is optimal, which makes it the safe way to add guidance.

## Project structure

```
.
├── 01_random_agent.py     # Random actions, shows the 8-number observation and crashes
├── 02_train_baseline.py   # Trains PPO on the built-in reward
├── 03_evaluate.py         # Scores a model on the plain env and records 3 videos
├── 04_train_hack.py       # Trains PPO with the exploitable hover bonus
├── 05_train_fix.py        # Trains PPO with the improvement-based reward
├── 06_watch.py            # Watch a saved model live in a window
├── wrappers.py            # TimePenaltyWrapper, HoverBonusWrapper, ImprovementBonusWrapper
├── models/                # Saved models (.zip)
├── videos/                # Recorded evaluation clips
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Or install directly:

```bash
pip install "gymnasium[box2d]" stable-baselines3 moviepy
```

If the Box2D install fails with an error mentioning `swig`, install swig first and then reinstall `gymnasium[box2d]`.

## How to run

Run from the project root with the virtual environment active. Create the output folders once:

```bash
mkdir models videos
```

```bash
# 1. Watch a random agent crash
python 01_random_agent.py

# 2. Train the baseline (about 15 to 40 minutes on a laptop CPU)
python 02_train_baseline.py
python 03_evaluate.py ppo_baseline

# 3. Train the broken reward and observe the hover
python 04_train_hack.py
python 03_evaluate.py ppo_hover_hack

# 4. Train the fixed reward
python 05_train_fix.py
python 03_evaluate.py ppo_fixed

# 5. Watch any model live
python 06_watch.py ppo_fixed
```

Each evaluation prints the mean score over 20 episodes and saves 3 clips to `videos/<model_name>/`.

## Notes and lessons learned

- **Judge by the video, not just the score.** A training reward can climb while the agent does something useless.
- **Always evaluate without your wrapper.** Otherwise your own bonus inflates the score.
- **Training is random.** Run more than one seed before calling a reward good or bad. The hack may take a retrain to appear.
- **Use `LunarLander-v3`.** Older tutorials reference `v2`, which no longer exists in current Gymnasium.
- **CPU is faster than GPU here.** The network is tiny, so `device="cpu"` with parallel environments (`SubprocVecEnv`) trains faster.

## Tech stack

Python, Gymnasium (Box2D), Stable-Baselines3 (PPO), PyTorch, MoviePy.

## References

- [Gymnasium: Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Stable-Baselines3 documentation](https://stable-baselines3.readthedocs.io/)
- [Hugging Face Deep RL Course, Unit 1](https://huggingface.co/learn/deep-rl-course/unit1/introduction)
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Ng, Harada, Russell, "Policy Invariance Under Reward Transformations" (1999), the basis for potential-based shaping
