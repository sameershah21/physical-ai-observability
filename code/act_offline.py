"""Shared helpers for testing an ACT policy offline, without the robot."""
import glob
import subprocess

import numpy as np
import pandas as pd
import torch
from PIL import Image
from safetensors.torch import load_file

from lerobot.policies.act.modeling_act import ACTPolicy


class OfflinePolicy:
    """Loads a saved policy with its normalisation statistics and predicts action chunks on the CPU."""

    def __init__(self, policy_dir):
        torch.manual_seed(0)
        self.policy = ACTPolicy.from_pretrained(policy_dir)
        self.policy.to("cpu").eval()
        self.policy.config.device = "cpu"
        self.img_key = list(self.policy.config.image_features)[0]
        self.chunk_size = int(self.policy.config.chunk_size)
        norm = load_file(glob.glob(policy_dir + "/policy_preprocessor_step_*_normalizer_processor.safetensors")[0])
        unnorm = load_file(glob.glob(policy_dir + "/policy_postprocessor_step_*_unnormalizer_processor.safetensors")[0])
        self.s_mean, self.s_std = norm["observation.state.mean"].float(), norm["observation.state.std"].float()
        self.a_mean, self.a_std = unnorm["action.mean"].float(), unnorm["action.std"].float()
        self.i_mean = norm[self.img_key + ".mean"].float().view(3, 1, 1)
        self.i_std = norm[self.img_key + ".std"].float().view(3, 1, 1)

    def predict(self, image, state):
        """image: float tensor (3, h, w) in [0, 1]; state: joint positions. Returns the chunk in joint units."""
        batch = {
            "observation.state": ((torch.as_tensor(state).float() - self.s_mean) / (self.s_std + 1e-8)).unsqueeze(0),
            self.img_key: ((image - self.i_mean) / (self.i_std + 1e-8)).unsqueeze(0),
        }
        with torch.no_grad():
            chunk = self.policy.predict_action_chunk(batch)[0]
        return (chunk * self.a_std + self.a_mean).numpy()


def load_image(path):
    return torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1).float() / 255.0


def extract_frames(dataset_dir, stride, tmp, tag):
    """Decodes every stride-th frame of the dataset's video with ffmpeg; returns the image paths in order."""
    video = sorted(glob.glob(dataset_dir + "/videos/*/chunk-*/*.mp4"))[0]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", video, "-vf", f"select='not(mod(n\\,{stride}))'", "-fps_mode", "vfr", f"{tmp}/{tag}_%05d.png"],
        check=True,
    )
    return sorted(glob.glob(f"{tmp}/{tag}_*.png"))


def table(dataset_dir):
    """Returns joint states, actions and episode index for every frame of the dataset."""
    df = pd.read_parquet(sorted(glob.glob(dataset_dir + "/data/chunk-*/*.parquet"))[0])
    stack = lambda c: np.stack(df[c].to_numpy())
    return stack("observation.state"), stack("action"), df["episode_index"].to_numpy()


def recorded_chunk(action, episode, n, chunk_size):
    """What was recorded over the chunk_size steps from frame n, padded with NaN past the end of the episode."""
    target = np.full((chunk_size, action.shape[1]), np.nan, dtype=np.float32)
    end = min(n + chunk_size, len(action))
    same = episode[n:end] == episode[n]
    target[: int(same.sum())] = action[n:end][same]
    return target
