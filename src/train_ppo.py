# src/train_ppo.py
import argparse
import os
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import FlappyBirdEnv

def make_env(pipe_gap: int, pixel_obs: bool):
    def _f():
        return FlappyBirdEnv(pipe_gap=pipe_gap, pixel_obs=pixel_obs)
    return _f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200000)
    ap.add_argument("--out", type=str, default="models/flappy_ppo.zip")
    ap.add_argument("--pipe_gap", type=int, default=110)
    ap.add_argument("--pixel_obs", action="store_true")
    args = ap.parse_args()

    env = DummyVecEnv([make_env(args.pipe_gap, args.pixel_obs)])
    policy = "CnnPolicy" if args.pixel_obs else "MlpPolicy"
    model = PPO(policy, env, verbose=1, n_steps=1024, batch_size=256, learning_rate=3e-4, gamma=0.99)
    model.learn(total_timesteps=args.timesteps)
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()