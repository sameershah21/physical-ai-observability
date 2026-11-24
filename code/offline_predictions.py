"""Offline check of an ACT policy: what does it output on its own training frames,
and on the frames of the recorded evaluation run? No robot is involved.

Run with the Python environment that has LeRobot installed, e.g.
    <solo-cli>/.venv/bin/python offline_predictions.py <policy_dir> <train_dataset_dir> <eval_dataset_dir> <out.npz> [train_stride]

Every 75th training frame is tested unless train_stride says otherwise, and every 100th frame of the
evaluation run.

<policy_dir> is a folder with config.json, model.safetensors and the saved normaliser files.

The .npz holds, for the sampled frames of each split: the frame index, the joint state, the
predicted chunk (frames x chunk x joints) and what was actually recorded over the same steps
(the demonstrator's actions for the training split, the commands sent for the evaluation
split), padded with NaN past the end of an episode.
"""
import sys
import tempfile

import numpy as np

from act_offline import OfflinePolicy, extract_frames, load_image, recorded_chunk, table

policy_dir, train_dir, eval_dir, out_path = sys.argv[1:5]
TRAIN_STRIDE, EVAL_STRIDE = (int(sys.argv[5]) if len(sys.argv) > 5 else 75), 100

policy = OfflinePolicy(policy_dir)
out = {"chunk_size": policy.chunk_size}
with tempfile.TemporaryDirectory() as tmp:
    for split, d, stride in (("train", train_dir, TRAIN_STRIDE), ("eval", eval_dir, EVAL_STRIDE)):
        state, action, episode = table(d)
        idx, preds, targets = [], [], []
        for k, path in enumerate(extract_frames(d, stride, tmp, split)):
            n = k * stride
            if n >= len(state):
                break
            idx.append(n)
            preds.append(policy.predict(load_image(path), state[n]))
            targets.append(recorded_chunk(action, episode, n, policy.chunk_size))
        idx = np.array(idx)
        out[f"{split}_frame"] = idx
        out[f"{split}_episode"] = episode[idx]
        out[f"{split}_state"] = state[idx].astype(np.float32)
        out[f"{split}_pred"] = np.stack(preds).astype(np.float32)
        out[f"{split}_recorded"] = np.stack(targets)
        mae = np.nanmean(np.abs(out[f"{split}_pred"] - out[f"{split}_recorded"]))
        print(f"{split}: {len(idx)} frames | mean abs difference from what was recorded: {mae:.2f}", flush=True)
np.savez_compressed(out_path, **out)
print("wrote", out_path)
