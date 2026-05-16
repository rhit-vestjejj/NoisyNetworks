"""
Standard Atari preprocessing for the DQN family (Mnih et al. 2015), used by
the NoisyNet paper for its 57-game evaluation.

Pipeline (per the paper / DQN-Nature):
  - NoOp resets (random no-op count at episode start, up to 30)
  - Frame skip + max over last 2 frames (action repeat 4)
  - 84 x 84 grayscale
  - Reward clipping to {-1, 0, +1}
  - 4-frame stacking
  - Episodic life (terminal on life loss for training only)
  - Fire on reset (some games need FIRE to start)

Gymnasium ships AtariPreprocessing + FrameStackObservation that cover almost
all of this; we add EpisodicLife + ClipReward on top.

Requires: pip install "gymnasium[atari,accept-rom-license]" ale-py
(do this on the server, not on the laptop).
"""

import gymnasium as gym
import numpy as np

try:
    import ale_py
except ImportError:
    ale_py = None
else:
    gym.register_envs(ale_py)


class EpisodicLifeEnv(gym.Wrapper):
    """End the episode when a life is lost (better learning signal).
    Only true reset on actual game-over (paper / DQN-Nature trick)."""

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Press FIRE on reset for envs that need it (Breakout, etc.)."""

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            obs, _ = self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            obs, _ = self.env.reset(**kwargs)
        return obs, {}


class ClipRewardEnv(gym.RewardWrapper):
    """Clip rewards to {-1, 0, +1} (DQN-Nature)."""

    def reward(self, reward):
        return float(np.sign(reward))


def make_atari(env_id, seed=0, episode_life=True, clip_rewards=True):
    """Build a fully-wrapped Atari env. env_id like 'ALE/Breakout-v5'."""
    env = gym.make(env_id, frameskip=1)  # frameskip handled by AtariPreprocessing
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,  # we handle this via EpisodicLifeEnv
        grayscale_obs=True,
        scale_obs=False,
    )
    if episode_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    env.action_space.seed(seed)
    return env
