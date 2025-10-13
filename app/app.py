# app/app.py

from pathlib import Path
import sys

# --- Make "src" importable ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# -----------------------------

import numpy as np
import streamlit as st
from PIL import Image

from env import FlappyBirdEnv

# Optional PPO imports (used only if model exists / toggle enabled)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Paths / globals
MODEL_PATH = ROOT / "models" / "flappy_ppo_v2.zip"

st.set_page_config(page_title="Flappy RL", page_icon="🐤")

st.title("Flappy RL — Human & Agent")
st.write("Play yourself or let a trained PPO agent fly (low-dim observations).")

# ----- Sidebar controls (define and persist early) -----
gap = st.sidebar.slider("Pipe gap", 80, 160, st.session_state.get("gap", 110), 5)
speed = st.sidebar.slider("Steps per tick", 1, 20, st.session_state.get("speed", 5))
auto = st.sidebar.checkbox("Auto-run", value=st.session_state.get("auto", False))
pixel_obs = st.sidebar.checkbox("Pixel observations", value=st.session_state.get("pix", False))
use_ppo = st.sidebar.checkbox("Use trained PPO (if available)", value=st.session_state.get("use_ppo", False))

# Persist to session state
st.session_state.gap = gap
st.session_state.speed = speed
st.session_state.auto = auto
st.session_state.pix = pixel_obs
st.session_state.use_ppo = use_ppo
# -------------------------------------------------------


# ----- (Re)create environment if needed -----
env_key = (gap, pixel_obs)
if "env" not in st.session_state or st.session_state.get("_env_key") != env_key:
    st.session_state.env = FlappyBirdEnv(pipe_gap=gap, pixel_obs=pixel_obs)
    st.session_state.obs, _ = st.session_state.env.reset()
    st.session_state.done = False
    st.session_state.score = 0
    st.session_state._env_key = env_key

env = st.session_state.env
# -------------------------------------------


# ----- Load PPO model once (cached in session) -----
if "ppo_model" not in st.session_state and MODEL_PATH.exists():
    try:
        # IMPORTANT: PPO here expects low-dimensional observations (shape (4,))
        pg = gap  # capture current gap value
        def _make_env():
            return FlappyBirdEnv(pipe_gap=pg, pixel_obs=False)
        venv = DummyVecEnv([_make_env])
        st.session_state.ppo_model = PPO.load(str(MODEL_PATH), env=venv, device="cpu")
        st.sidebar.success("PPO model loaded.")
    except Exception as e:
        st.sidebar.warning(f"Could not load PPO model: {e}")

ppo_model = st.session_state.get("ppo_model", None)

# Warn if user tries PPO with pixel obs
if use_ppo and pixel_obs:
    st.sidebar.warning("PPO expects low-dim observations. Turn OFF 'Pixel observations' to use the model.")


# ----- Controls row -----
col_reset, col_flap, col_step = st.columns(3)
if col_reset.button("Reset"):
    st.session_state.obs, _ = env.reset()
    st.session_state.done = False
    st.session_state.score = 0

flap = col_flap.button("FLAP (human)")
step_once = col_step.button("Step")
# ------------------------


# ----- Policy function -----
def policy(obs: np.ndarray) -> int:
    # 1) Trained PPO (only for low-dim obs)
    if (
        use_ppo
        and ppo_model is not None
        and isinstance(obs, np.ndarray)
        and obs.shape == (4,)
        and not pixel_obs
    ):
        action, _ = ppo_model.predict(obs, deterministic=True)
        return int(action)

    # 2) Human override
    if flap:
        return 1

    # 3) Heuristic fallback (works even without PPO)
    if isinstance(obs, np.ndarray) and obs.ndim == 1 and obs.shape[0] == 4:
        bird_y, bird_vel, next_dx, gap_y = obs
        return int(bird_y > gap_y)

    # 4) Random last resort
    return int(np.random.rand() < 0.1)
# --------------------------


# ----- Step loop -----
steps = speed if (auto or step_once or flap) else 0
for _ in range(steps):
    if st.session_state.done:
        break
    # If pixel_obs is ON, the observation is an image; policy should not use PPO then.
    current_obs = st.session_state.obs if not pixel_obs else np.array([0, 0, 0, 0], dtype=np.float32)
    action = policy(current_obs)
    obs, r, term, trunc, info = env.step(action)
    st.session_state.obs = obs
    st.session_state.done = term or trunc
    st.session_state.score = info.get("score", st.session_state.score)
# ---------------------


# ----- Debug (obs & action) -----
with st.sidebar.expander("Debug (obs & action)"):
    if isinstance(st.session_state.obs, np.ndarray):
        try:
            st.write("obs:", np.round(st.session_state.obs, 2).tolist())
        except Exception:
            st.write("obs:", "array (non-serializable)")
        if pixel_obs:
            st.caption("Pixel observations are ON. PPO path is disabled unless you trained with pixels.")
    else:
        st.write("obs: N/A")

    if (
        use_ppo
        and ppo_model is not None
        and isinstance(st.session_state.obs, np.ndarray)
        and st.session_state.obs.shape == (4,)
        and not pixel_obs
    ):
        act, _ = ppo_model.predict(st.session_state.obs, deterministic=True)
        st.write("ppo_action (debug):", int(act))
# --------------------------------


# ----- Render -----
frame = env.render()
st.image(
    Image.fromarray(frame),
    caption=f"Score: {st.session_state.score}  |  {'DONE' if st.session_state.done else 'RUNNING'}"
)
st.caption("Tip: Toggle 'Use trained PPO' to watch your agent fly. Train with: python src/train_ppo.py --timesteps 200000 --out models/flappy_ppo.zip")
# ---------------
