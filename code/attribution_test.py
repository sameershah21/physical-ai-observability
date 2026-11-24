"""Two more checks on a single-camera ACT policy, from recordings alone.

    <solo-cli>/.venv/bin/python attribution_test.py <policy_dir> <train_dataset_dir> <unseen_dataset_dir> <rollout_dataset_dir> <out.npz> [attribution-only]

With "attribution-only" as a last argument only the first check is run (for tracking it over the
checkpoints of one training run).

1. Which input does an answer come from: the camera, or the joint readings?
   Take two recorded frames i and j. The policy is asked four times: with the picture and the joint
   readings of i, with the picture of j and the joints of i, with the picture of i and the joints of
   j, and with both of j. For two inputs the Shapley value has a closed form: each input's share is
   the change it causes, averaged over whether the other input has already been swapped. The two
   shares add up exactly to the difference between the two original answers.

2. How far is a new picture from the pictures the policy was trained on?
   Each frame is passed through the policy's own vision backbone and averaged over the picture to
   one vector. The distance from a new frame's vector to the nearest training frame's vector is
   stored, with the frame index of that neighbour so that neighbours close in time can be left out
   when training frames are compared with each other.

Frames are taken every STRIDE-th step. Pairs (i, j) are drawn with a fixed seed from frames at
least MIN_GAP steps apart.
"""
import sys
import tempfile

import numpy as np
import torch

from act_offline import OfflinePolicy, extract_frames, load_image, table

policy_dir, train_dir, unseen_dir, rollout_dir, out_path = sys.argv[1:6]
ATTRIBUTION_ONLY = len(sys.argv) > 6 and sys.argv[6] == "attribution-only"
STRIDE, PAIRS, MIN_GAP = 30, 60, 150
GAINS = [1.25, 2.0]

policy = OfflinePolicy(policy_dir)
captured = {}
policy.policy.model.backbone.register_forward_hook(lambda module, inputs, output: captured.__setitem__("map", output["feature_map"]))


def predict_with_feature(image, state):
    chunk = policy.predict(image, state)
    return chunk, captured["map"].mean(dim=(2, 3))[0].numpy().copy()


out = {"chunk_size": policy.chunk_size, "stride": STRIDE, "gains": np.array(GAINS)}
frames = {}
with tempfile.TemporaryDirectory() as tmp:
    for split, d in (("train", train_dir), ("unseen", unseen_dir), ("rollout", rollout_dir)):
        if ATTRIBUTION_ONLY and split != "train":
            continue
        state, action, episode = table(d)
        idx, images, feats, chunks = [], [], [], []
        bright_feats = []
        for k, path in enumerate(extract_frames(d, STRIDE, tmp, split)):
            n = k * STRIDE
            if n >= len(state):
                break
            image = load_image(path)
            chunk, feat = predict_with_feature(image, state[n])
            idx.append(n), feats.append(feat), chunks.append(chunk)
            if split == "train":
                images.append(image)
                if not ATTRIBUTION_ONLY:
                    bright_feats.append([predict_with_feature((image * g).clamp(0, 1), state[n])[1] for g in GAINS])
        idx = np.array(idx)
        out[f"{split}_frame"], out[f"{split}_feature"] = idx, np.stack(feats).astype(np.float32)
        out[f"{split}_pred"] = np.stack(chunks).astype(np.float32)
        if split == "train":
            if not ATTRIBUTION_ONLY:
                out["train_bright_feature"] = np.array(bright_feats, dtype=np.float32)  # frames x gains x channels
            frames = {"idx": idx, "images": images, "state": state[idx]}
        print(f"{split}: {len(idx)} frames", flush=True)

# --- which input an answer comes from, on pairs of training frames
rng = np.random.default_rng(0)
idx, images, states = frames["idx"], frames["images"], frames["state"]
pairs = []
while len(pairs) < PAIRS:
    i, j = rng.integers(0, len(idx), size=2)
    if abs(int(idx[i]) - int(idx[j])) >= MIN_GAP:
        pairs.append((int(i), int(j)))
own = out["train_pred"]
phi_image, phi_state, total = [], [], []
for n, (i, j) in enumerate(pairs):
    f_ii, f_jj = own[i], own[j]
    f_ji = policy.predict(images[j], states[i])  # picture of j, joints of i
    f_ij = policy.predict(images[i], states[j])  # picture of i, joints of j
    phi_image.append(0.5 * ((f_ii - f_ji) + (f_ij - f_jj)))
    phi_state.append(0.5 * ((f_ii - f_ij) + (f_ji - f_jj)))
    total.append(f_ii - f_jj)
    if (n + 1) % 20 == 0:
        print(f"pairs: {n + 1} of {PAIRS}", flush=True)
out["pairs"] = np.array(pairs)
out["phi_image"], out["phi_state"], out["total"] = (np.stack(v).astype(np.float32) for v in (phi_image, phi_state, total))
size = lambda v: float(np.abs(np.stack(v)).mean())
camera, joints = size(phi_image), size(phi_state)
print(f"answers differ by {size(total):.2f} between the two frames of a pair | from the camera {camera:.2f} | from the joint readings {joints:.2f}"
      f" | camera share {100 * camera / (camera + joints + 1e-9):.0f}%", flush=True)
np.savez_compressed(out_path, **out)
print("wrote", out_path)
