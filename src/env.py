# src/env.py
import random
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from PIL import Image, ImageDraw
import gymnasium as gym
from gymnasium import spaces

class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    def __init__(self, render_mode: Optional[str]=None, pixel_obs: bool=False, seed: int=42,
                 screen_size: Tuple[int,int]=(288,512), pipe_gap: int=110, pipe_distance: int=180,
                 gravity: float=0.35, flap_velocity: float=-6.5, pipe_speed: float=2.5, bird_radius: int=10):
        super().__init__()
        self.rng = random.Random(seed)
        self.W, self.H = screen_size
        self.pipe_gap=pipe_gap; self.pipe_distance=pipe_distance
        self.gravity=gravity; self.flap_velocity=flap_velocity; self.pipe_speed=pipe_speed
        self.bird_r=bird_radius; self.pixel_obs=pixel_obs; self.render_mode=render_mode
        if pixel_obs:
            self.observation_space = spaces.Box(0,255, shape=(self.H,self.W,3), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=np.array([0, -20, 0, 0],dtype=np.float32),
                                                high=np.array([self.H, 20, self.W, self.H],dtype=np.float32),
                                                dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._reset_state()

    def _reset_state(self):
        self.bird_y = self.H//2; self.bird_vel=0.0; self.score=0; self.t=0
        self.pipes: List[Dict[str,float]] = []
        x = self.W + 30
        for _ in range(3):
            self.pipes.append({"x": x, "gap_y": self.rng.randint(100, self.H-100)}); x += self.pipe_distance

    def bird_x(self): return int(self.W*0.2)
    def _next_pipe(self):
        bx=self.bird_x(); pipe_w=52
        fut=[p for p in self.pipes if p["x"]+pipe_w>=bx-1]
        return fut[0] if fut else self.pipes[0]

    def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None):
        if seed is not None: self.rng.seed(seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action: int):
        self.t += 1
        if action==1: self.bird_vel = self.flap_velocity
        self.bird_vel += self.gravity
        self.bird_y += self.bird_vel
        for p in self.pipes: p["x"] -= self.pipe_speed
        if self.pipes and self.pipes[0]["x"] < -60:
            self.pipes.pop(0)
            self.pipes.append({"x": self.pipes[-1]["x"]+self.pipe_distance,
                               "gap_y": self.rng.randint(100, self.H-100)})
        reward = 0.1; terminated=False
        pipe=self._next_pipe(); pipe_center_x = pipe["x"]+26

        # Encourage the bird to stay near the gap center
        dy = abs(self.bird_y - pipe["gap_y"])
        reward += 0.002 * (1.0 - min(dy / (self.H / 2), 1.0))

        if pipe_center_x < self.bird_x() <= pipe_center_x + self.pipe_speed:
            self.score += 1; reward += 1.0
        if self._collide():
            reward -= 1.0; terminated=True
        return self._obs(), reward, terminated, False, {"score": self.score, "t": self.t}

    def _collide(self)->bool:
        if self.bird_y - self.bird_r <= 0 or self.bird_y + self.bird_r >= self.H: return True
        bx=self.bird_x(); by=self.bird_y; r=self.bird_r; pipe_w=52; gap=self.pipe_gap
        for p in self.pipes:
            x=int(p["x"]); gy=int(p["gap_y"]); top_h=int(gy-gap/2); bot_y=int(gy+gap/2)
            if self._circle_rect(bx,by,r,x,0,pipe_w,top_h): return True
            if self._circle_rect(bx,by,r,x,bot_y,pipe_w,self.H-bot_y): return True
        return False

    @staticmethod
    def _circle_rect(cx,cy,cr,rx,ry,rw,rh)->bool:
        closest_x = min(max(cx, rx), rx+rw); closest_y = min(max(cy, ry), ry+rh)
        dx=cx-closest_x; dy=cy-closest_y
        return dx*dx + dy*dy <= cr*cr

    def _obs(self):
        """Return a normalized observation vector [bird_y, bird_vel, pipe_dx, pipe_gap_y]."""
        if self.pixel_obs:
            return self.render()

        n = self._next_pipe()

        # Normalize values to make training easier and more stable
        by = self.bird_y / self.H                      # 0–1
        bv = np.tanh(self.bird_vel / 10.0)             # ~-1..1
        dx = (n["x"] - self.bird_x()) / self.W         # 0–1
        gy = n["gap_y"] / self.H                       # 0–1

        return np.array([by, bv, dx, gy], dtype=np.float32)

    def render(self):
        img = Image.new("RGB", (self.W, self.H), (135, 206, 235))
        d = ImageDraw.Draw(img)
        d.rectangle([(0, self.H - 70), (self.W, self.H)], fill=(222, 216, 149))
        pipe_w = 52
        for p in self.pipes:
            x = int(p["x"])
            gy = int(p["gap_y"])
            top_h = int(gy - self.pipe_gap / 2)
            bot_y = int(gy + self.pipe_gap / 2)
            d.rectangle([(x, 0), (x + pipe_w, top_h)], fill=(70, 200, 70))
            d.rectangle([(x, bot_y), (x + pipe_w, self.H)], fill=(70, 200, 70))
        bx = self.bird_x()
        by = int(self.bird_y)
        r = self.bird_r
        d.ellipse([(bx - r, by - r), (bx + r, by + r)], fill=(255, 215, 0), outline=(0, 0, 0))
        return np.array(img, dtype=np.uint8)


