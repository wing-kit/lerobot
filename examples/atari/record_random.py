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

import argparse
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from examples.atari.env import make_atari_env


def main():
    parser = argparse.ArgumentParser(description="Record random Atari gameplay into a LeRobotDataset")
    parser.add_argument("--task", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--repo_id", type=str, default="local/atari_breakout_random")
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--grayscale", action="store_true")
    args = parser.parse_args()

    env = make_atari_env(args.task, grayscale=args.grayscale)
    obs, _ = env.reset()

    h, w, c = obs["pixels"].shape
    features = {
        "pixels": {"dtype": "video", "shape": (h, w, c), "names": ["h", "w", "c"]},
        "action": {"dtype": "int64", "shape": (1,), "names": None},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=env.metadata.get("render_fps", 30),
        root=args.root,
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
        features=features,
    )

    for ep in range(args.episodes):
        obs, _ = env.reset()
        for t in range(args.max_steps):
            action = np.array([env.action_space.sample()], dtype=np.int64)
            next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            frame = {
                "pixels": obs["pixels"],
                "action": action,
                "next.reward": np.array([reward], dtype=np.float32),
                "next.done": np.array([terminated or truncated], dtype=bool),
            }
            ds.add_frame(frame)
            obs = next_obs
            if terminated or truncated:
                ds.save_episode()
                break

    env.close()
    print(f"Saved {args.episodes} episodes to {args.root}/{args.repo_id}")


if __name__ == "__main__":
    main()

