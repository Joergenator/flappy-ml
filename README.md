# Flappy RL Starter
A ready-to-run Flappy Bird clone tailored for reinforcement learning.

- **Gymnasium-like environment** (`src/env.py`) with reset/step/render
- **Discrete actions:** 0=no-op, 1=flap
- **Reward:** +0.1 alive, +1 pass pipe, -1 on death
- **Streamlit UI** (`app/app.py`) for human play or watching an agent
- **Training script** (`src/train_ppo.py`) for PPO with stable-baselines3

## Quickstart
Link til program:
https://flappy-ml-nkcoabwarmh3zj7hmqpphd.streamlit.app/ 

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
streamlit run app/app.py
```

## Train PPO
```bash
python src/train_ppo.py --timesteps 200000 --out models/flappy_ppo.zip
```