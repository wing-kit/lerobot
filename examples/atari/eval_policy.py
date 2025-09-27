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
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors

from examples.atari.env import make_atari_env


def main():
    parser = argparse.ArgumentParser(description="Evaluate a LeRobot policy on Atari with Gymnasium")
    parser.add_argument("--policy_path", type=str, required=True, help="Path or repo id of pretrained policy")
    parser.add_argument("--task", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    env = make_atari_env(args.task)

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy = make_policy(cfg=policy_cfg)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg, pretrained_path=policy_cfg.pretrained_path
    )

    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_rew = 0.0
        with torch.inference_mode():
            while True:
                # Map observation to LeRobot expected dict (VanillaObservationProcessor handles 'pixels')
                proc_obs = preprocessor({"observation.pixels": obs["pixels"]})
                action = policy.select_action(proc_obs)
                action = postprocessor(action)

                # For discrete heads, action may be logits or integer. Try to coerce to int.
                act = action
                if hasattr(act, "softmax"):
                    act = act.softmax(dim=-1).argmax(dim=-1)
                act = int(np.array(act.detach().cpu()).squeeze())

                obs, reward, terminated, truncated, _ = env.step(act)
                ep_rew += reward
                if terminated or truncated:
                    break
        print(f"Episode {ep}: reward={ep_rew}")

    env.close()


if __name__ == "__main__":
    main()

