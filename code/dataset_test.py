"""Tests an ACT policy on demonstrations it was trained on and on demonstrations it never saw,
and repeats the test with the camera images brightened. Runs on the training machine.

    python dataset_test.py <policy_dir> <dataset_repo_id> <train_episodes> <held_out_episodes> <out.npz>
    e.g.   dataset_test.py outputs/train/x/checkpoints/020000/pretrained_model sameer-s1/bread-in-toaster-dataset-v1 0-57:5 58-63 out.npz

Episodes are given as ranges; "0-57:5" means every fifth episode of 0..57. Every STRIDE-th frame
of each chosen episode is tested. For each tested frame the .npz holds the predicted chunk at
every brightness gain, what the demonstrators did over the same steps (NaN past the end of the
episode), and the share of pixels at or near white.
"""
import glob
import sys

import numpy as np
import torch
from safetensors.torch import load_file

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy

policy_dir, repo_id, train_spec, held_spec, out_path = sys.argv[1:6]
STRIDE = 30
GAINS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
device = "cuda" if torch.cuda.is_available() else "cpu"


def episodes(spec):
    rng, _, step = spec.partition(":")
    lo, _, hi = rng.partition("-")
    return list(range(int(lo), int(hi or lo) + 1, int(step or 1)))


torch.manual_seed(0)
policy = ACTPolicy.from_pretrained(policy_dir)
policy.config.device = device
policy.to(device).eval()
H = int(policy.config.chunk_size)
image_keys = list(policy.config.image_features)
norm = load_file(glob.glob(policy_dir + "/policy_preprocessor_step_*_normalizer_processor.safetensors")[0])
unnorm = load_file(glob.glob(policy_dir + "/policy_postprocessor_step_*_unnormalizer_processor.safetensors")[0])
dev = lambda t: t.float().to(device)
s_mean, s_std = dev(norm["observation.state.mean"]), dev(norm["observation.state.std"])
a_mean, a_std = dev(unnorm["action.mean"]), dev(unnorm["action.std"])
i_stats = {k: (dev(norm[k + ".mean"]).view(3, 1, 1), dev(norm[k + ".std"]).view(3, 1, 1)) for k in image_keys}


def predict(images, state):
    batch = {"observation.state": ((state - s_mean) / (s_std + 1e-8)).unsqueeze(0)}
    for k in image_keys:
        m, sd = i_stats[k]
        batch[k] = ((images[k] - m) / (sd + 1e-8)).unsqueeze(0)
    with torch.no_grad():
        chunk = policy.predict_action_chunk(batch)[0]
    return (chunk * a_std + a_mean).cpu().numpy()


train_eps, held_eps = episodes(train_spec), episodes(held_spec)
fps = LeRobotDataset(repo_id, episodes=[held_eps[0]]).fps
ds = LeRobotDataset(repo_id, episodes=sorted(train_eps + held_eps), delta_timestamps={"action": [i / fps for i in range(H)]})
print(f"policy {policy_dir} | chunk {H} | cameras {image_keys} | {len(ds)} frames in {len(train_eps)} trained-on and {len(held_eps)} held-out episodes", flush=True)

rows = {"episode": [], "frame": [], "held_out": [], "state": [], "pred": [], "recorded": [], "saturated": []}
for i in range(0, len(ds), STRIDE):
    item = ds[i]
    ep = int(item["episode_index"])
    state = dev(item["observation.state"])
    images = {k: dev(item[k]) for k in image_keys}
    target = item["action"].numpy().astype(np.float32)
    target[item["action_is_pad"].numpy().astype(bool)] = np.nan
    per_gain, sat = [], []
    for g in GAINS:
        bright = {k: (v * g).clamp(0, 1) for k, v in images.items()}
        per_gain.append(predict(bright, state))
        sat.append(float(np.mean([(b.mean(0) >= 250 / 255).float().mean().item() for b in bright.values()])))
    rows["episode"].append(ep), rows["frame"].append(int(item["frame_index"])), rows["held_out"].append(ep in held_eps)
    rows["state"].append(state.cpu().numpy()), rows["pred"].append(np.stack(per_gain)), rows["recorded"].append(target), rows["saturated"].append(sat)

out = {k: np.array(v) for k, v in rows.items()}
out["pred"], out["state"] = out["pred"].astype(np.float32), out["state"].astype(np.float32)
out["gains"], out["chunk_size"] = np.array(GAINS), H
np.savez_compressed(out_path, **out)
err = np.nanmean(np.abs(out["pred"][:, 0] - out["recorded"]), axis=(1, 2))
held = out["held_out"]
print(f"tested frames: {len(err)} | error on trained-on episodes {err[~held].mean():.2f} | on held-out episodes {err[held].mean():.2f}", flush=True)
for gi, g in enumerate(GAINS):
    e = np.nanmean(np.abs(out["pred"][:, gi] - out["recorded"]), axis=(1, 2))
    print(f"  gain {g:4.2f} | near white {100 * out['saturated'][:, gi].mean():5.1f}% | error trained-on {e[~held].mean():6.2f} | held-out {e[held].mean():6.2f}", flush=True)
print("wrote", out_path)
