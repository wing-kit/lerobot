#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from __future__ import annotations

import gymnasium as gym
import numpy as np


def make_atari_env(task: str = "ALE/Breakout-v5", grayscale: bool = False, frame_size: tuple[int, int] = (84, 84)):
    """Create an Atari Gymnasium environment that exposes observations under the key 'pixels'.

    The returned env has observations shaped as a dict: {"pixels": np.ndarray (H, W, C|1) uint8}

    Args:
        task: Gymnasium Atari task id, e.g. "ALE/Breakout-v5", "ALE/Pong-v5".
        grayscale: If True, apply grayscale wrapper and keep a channel dimension.
        frame_size: Resize frames to this (H, W).

    Returns:
        Gymnasium environment whose observation is a dict with a single key 'pixels'.
    """
    env = gym.make(task, render_mode="rgb_array")

    # Optional wrappers (if available)
    try:
        from gymnasium.wrappers import ResizeObservation, GrayScaleObservation, TransformObservation
    except Exception:  # pragma: no cover - fallback path if wrappers are unavailable
        ResizeObservation = GrayScaleObservation = TransformObservation = None

    if grayscale and GrayScaleObservation is not None:
        env = GrayScaleObservation(env, keep_dim=True)

    if ResizeObservation is not None:
        env = ResizeObservation(env, frame_size)

    if TransformObservation is not None:
        # Map raw observation -> {"pixels": uint8 HWC}
        def to_pixels(obs: np.ndarray) -> dict[str, np.ndarray]:
            arr = obs if obs.dtype == np.uint8 else obs.astype(np.uint8)
            return {"pixels": arr}

        obs_space = env.observation_space
        assert isinstance(obs_space, gym.spaces.Box), "Expected Box observation space for Atari envs"
        dict_space = gym.spaces.Dict({
            "pixels": gym.spaces.Box(low=0, high=255, shape=obs_space.shape, dtype=np.uint8)
        })
        env = TransformObservation(env, to_pixels, observation_space=dict_space)
        return env

    # Fallback: custom wrapper to expose {"pixels": ...}
    class PixelsWrapper(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            obs_space = env.observation_space
            assert isinstance(obs_space, gym.spaces.Box), "Expected Box observation space for Atari envs"
            self.observation_space = gym.spaces.Dict({
                "pixels": gym.spaces.Box(low=0, high=255, shape=obs_space.shape, dtype=np.uint8)
            })

        def observation(self, obs):
            arr = obs if obs.dtype == np.uint8 else obs.astype(np.uint8)
            return {"pixels": arr}

    return PixelsWrapper(env)


if __name__ == "__main__":
    # Smoke test
    env = make_atari_env("ALE/Pong-v5")
    obs, _ = env.reset()
    assert "pixels" in obs
    for _ in range(5):
        obs, rew, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    env.close()

