from pathlib import Path
import sys, os
import numpy as np
import streamlit as st
from PIL import Image

# --- Make "src" importable ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from env import FlappyBirdEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

st.set_page_config(page_title="Flappy RL", page_icon="🐤")

st.title("Flappy RL — RL-only agent")
st.write("Watch a trained PPO agent (or baseline heuristic) play Flappy Bird.")

# ---------- Model selection ----------
models_dir = ROOT / "models"
models_dir.mkdir(exist_ok=True)

model_files = sorted(
    [f for f in os.listdir(models_dir) if f.endswith(".zip")],
    key=lambda f: os.path.getmtime(models_dir / f),
    reverse=True
)

prev = st.session_state.get("selected_model_name")
default_index = model_files.index(prev) if prev in model_files else 0

selected_model = st.sidebar.selectbox(
    "Select PPO model",
    model_files if model_files else ["(no models found)"],
    index=default_index if model_files else 0,
    disabled=(len(model_files) == 0)
)

if model_files:
    st.session_state.selected_model_name = selected_model
    MODEL_PATH = models_dir / selected_model
    st.sidebar.markdown(f"Using model: `{MODEL_PATH.name}`")
else:
    MODEL_PATH = None
    st.sidebar.warning("No models in /models — please train one.")

# ---------- Settings ----------
gap = st.sidebar.slider("Pipe gap", 80, 160, st.session_state.get("gap", 110), 5)
steps_per_tick = st.sidebar.slider("Steps per tick", 1, 30, st.session_state.get("speed", 5))
use_ppo = st.sidebar.checkbox("Use PPO model", value=st.session_state.get("use_ppo", False))

st.session_state.gap = gap
st.session_state.speed = steps_per_tick
st.session_state.use_ppo = use_ppo

# ---------- Environment setup ----------
env_key = gap
if "env" not in st.session_state or st.session_state.get("_env_key") != env_key:
    st.session_state.env = FlappyBirdEnv(pipe_gap=gap, pixel_obs=False)
    st.session_state.obs, _ = st.session_state.env.reset()
    st.session_state.score = 0
    st.session_state.done = False
    st.session_state._env_key = env_key

env = st.session_state.env

# ---------- Load PPO if needed ----------
ppo_model = st.session_state.get("ppo_model", None)
loaded_name = st.session_state.get("loaded_model_name")

if MODEL_PATH and (ppo_model is None or loaded_name != MODEL_PATH.name):
    try:
        def _make_env():
            return FlappyBirdEnv(pipe_gap=gap, pixel_obs=False)
        vec = DummyVecEnv([_make_env])

        new_model = PPO.load(str(MODEL_PATH), env=vec, device="cpu")
        st.session_state.ppo_model = new_model
        st.session_state.loaded_model_name = MODEL_PATH.name
        ppo_model = new_model
        st.sidebar.success("PPO model loaded")
    except Exception as e:
        st.session_state.ppo_model = None
        st.session_state.loaded_model_name = None
        ppo_model = None
        st.sidebar.error(f"Failed to load model: {e}")

# ---------- Controls ----------
col_reset, col_step = st.columns(2)
if col_reset.button("Reset"):
    st.session_state.obs, _ = env.reset()
    st.session_state.score = 0
    st.session_state.done = False

step_once = col_step.button("Run steps")

# ---------- Policy ----------
def policy(obs):
    if use_ppo and ppo_model is not None:
        action, _ = ppo_model.predict(obs, deterministic=True)
        return int(action)

    # Heuristic baseline: flap if below gap center
    bird_y, bird_vel, dx, gap_y = obs
    return int(bird_y > gap_y)

# ---------- Step loop ----------
steps = steps_per_tick if step_once else 0
for _ in range(steps):
    if st.session_state.done:
        break
    action = policy(st.session_state.obs)
    obs, r, term, trunc, info = env.step(action)

    st.session_state.obs = obs
    st.session_state.done = term or trunc
    st.session_state.score = info.get("score", st.session_state.score)

# ---------- Render ----------
frame = env.render()
st.image(Image.fromarray(frame),
         caption=f"Score: {st.session_state.score} | {'DEAD' if st.session_state.done else 'RUNNING'}")
