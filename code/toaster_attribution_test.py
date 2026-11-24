"""Which input does a two-camera ACT policy's answer come from: each camera, or the joint readings?

    <solo-cli>/.venv/bin/python toaster_attribution_test.py <policy_dir> <dataset_repo_id> <episodes e.g. 58-63> <out.npz>

For pairs of recorded frames (i, j), the policy is asked eight times: with every combination of
"this input from frame i, that input from frame j" over its three inputs (front camera, hand
camera, joint readings). From those eight answers the exact Shapley value of each input follows:
its share of the difference between the answer for frame i and the answer for frame j. The three
shares add up exactly to that difference.

Frames are taken every STRIDE-th step of the chosen episodes; pairs are drawn with a fixed seed
from frames at least MIN_GAP steps apart or in different episodes.
"""
import glob
import itertools
import sys

import numpy as np
import torch
from safetensors.torch import load_file

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy

policy_dir, repo_id, spec, out_path = sys.argv[1:5]
STRIDE, PAIRS, MIN_GAP = 30, 40, 150


def episodes(text):
    rng, _, step = text.partition(":")
    lo, _, hi = rng.partition("-")
    return list(range(int(lo), int(hi or lo) + 1, int(step or 1)))


torch.manual_seed(0)
policy = ACTPolicy.from_pretrained(policy_dir)
policy.config.device = "cpu"
policy.to("cpu").eval()
image_keys = list(policy.config.image_features)
norm = load_file(glob.glob(policy_dir + "/policy_preprocessor_step_*_normalizer_processor.safetensors")[0])
unnorm = load_file(glob.glob(policy_dir + "/policy_postprocessor_step_*_unnormalizer_processor.safetensors")[0])
s_mean, s_std = norm["observation.state.mean"].float(), norm["observation.state.std"].float()
a_mean, a_std = unnorm["action.mean"].float(), unnorm["action.std"].float()
i_stats = {k: (norm[k + ".mean"].float().view(3, 1, 1), norm[k + ".std"].float().view(3, 1, 1)) for k in image_keys}
players = image_keys + ["observation.state"]


def predict(inputs):
    """inputs: {image key: image in [0, 1], "observation.state": joint readings}. Returns the chunk in joint units."""
    batch = {"observation.state": ((inputs["observation.state"] - s_mean) / (s_std + 1e-8)).unsqueeze(0)}
    for k in image_keys:
        m, sd = i_stats[k]
        batch[k] = ((inputs[k] - m) / (sd + 1e-8)).unsqueeze(0)
    with torch.no_grad():
        chunk = policy.predict_action_chunk(batch)[0]
    return (chunk * a_std + a_mean).numpy()


ds = LeRobotDataset(repo_id, episodes=episodes(spec), video_backend="pyav")  # torchcodec does not load on every machine
items = []
for n in range(0, len(ds), STRIDE):
    item = ds[n]
    items.append({"episode": int(item["episode_index"]), "frame": int(item["frame_index"]),
                  **{k: item[k].float() for k in players}})
print(f"policy {policy_dir} | inputs {players} | {len(items)} frames from episodes {spec}", flush=True)

rng = np.random.default_rng(0)
pairs = []
while len(pairs) < PAIRS:
    i, j = (int(v) for v in rng.integers(0, len(items), size=2))
    if items[i]["episode"] != items[j]["episode"] or abs(items[i]["frame"] - items[j]["frame"]) >= MIN_GAP:
        pairs.append((i, j))

n_players = len(players)
subsets = [frozenset(c) for r in range(n_players + 1) for c in itertools.combinations(range(n_players), r)]
weight = {0: 1 / 3, 1: 1 / 6, 2: 1 / 3}  # |S|! (n - |S| - 1)! / n! for n = 3
assert n_players == 3, "the weights above are for three inputs"
phi, total = [], []
for count, (i, j) in enumerate(pairs):
    value = {s: predict({k: items[i if p in s else j][k] for p, k in enumerate(players)}) for s in subsets}
    shares = []
    for p in range(n_players):
        others = [s for s in subsets if p not in s]
        shares.append(sum(weight[len(s)] * (value[s | {p}] - value[s]) for s in others))
    phi.append(np.stack(shares))
    total.append(value[frozenset(range(n_players))] - value[frozenset()])
    if (count + 1) % 10 == 0:
        print(f"pairs: {count + 1} of {PAIRS}", flush=True)
phi, total = np.stack(phi).astype(np.float32), np.stack(total).astype(np.float32)  # pairs x inputs x chunk x joints
assert np.abs(phi.sum(1) - total).max() < 1e-2, "the shares must add up to the whole difference"
np.savez_compressed(out_path, phi=phi, total=total, pairs=np.array(pairs), players=np.array(players))
moving = slice(0, 6)  # the arm that moves in this task
size = np.abs(phi[..., moving]).mean((0, 2, 3))
print(f"answers differ by {np.abs(total[..., moving]).mean():.2f} between the two frames of a pair, on the moving arm", flush=True)
for k, v in zip(players, size):
    print(f"  from {k:32s}: {v:6.2f}  ({100 * v / size.sum():.0f}%)", flush=True)
print("wrote", out_path)
