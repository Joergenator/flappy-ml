# app/app.py

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


import numpy as np
import streamlit as st
from PIL import Image

from env import FlappyBirdEnv

st.set_page_config(page_title="Flappy RL", page_icon="🐤")

st.title("Flappy RL — Human & Agent")
st.write("Play or watch a simple policy. Bring your own trained PPO model later.")

# Sidebar
st.sidebar.header("Options")
gap = st.sidebar.slider("Pipe gap", 80, 160, 110, 5)
speed = st.sidebar.slider("Steps per tick", 1, 20, 5)
auto = st.sidebar.checkbox("Auto-run", value=False)
pixel_obs = st.sidebar.checkbox("Pixel observations", value=True)

# Setup env in session
if "env" not in st.session_state or st.session_state.get("gap") != gap or st.session_state.get("pix") != pixel_obs:
    st.session_state.gap = gap; st.session_state.pix = pixel_obs
    st.session_state.env = FlappyBirdEnv(pipe_gap=gap, pixel_obs=pixel_obs)
    st.session_state.obs, _ = st.session_state.env.reset()
    st.session_state.done = False
    st.session_state.score = 0

env = st.session_state.env

col_reset, col_flap, col_step = st.columns(3)
if col_reset.button("Reset"):
    st.session_state.obs, _ = env.reset()
    st.session_state.done=False; st.session_state.score=0

flap = col_flap.button("FLAP (human)")
step_once = col_step.button("Step")

def policy(obs):
    if flap: return 1
    # Heuristic controller using low-dim obs
    if isinstance(obs, np.ndarray) and obs.ndim==1 and obs.shape[0]==4:
        bird_y, bird_vel, next_dx, gap_y = obs
        return int(bird_y > gap_y)
    # random fallback
    return int(np.random.rand()<0.1)

steps = speed if (auto or step_once or flap) else 0
for _ in range(steps):
    if st.session_state.done: break
    a = policy(st.session_state.obs if not pixel_obs else np.array([0,0,0,0],dtype=np.float32))
    obs, r, term, trunc, info = env.step(a)
    st.session_state.obs = obs; st.session_state.done = term or trunc
    st.session_state.score = info.get("score", st.session_state.score)

frame = env.render()
st.image(Image.fromarray(frame), caption=f"Score: {st.session_state.score}  |  {'DONE' if st.session_state.done else 'RUNNING'}")
st.caption("Train with src/train_ppo.py to replace the heuristic with a real PPO policy.")