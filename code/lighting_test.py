"""How much do a policy's predictions change when the same training frames are overexposed?

    <solo-cli>/.venv/bin/python lighting_test.py <policy_dir> <train_dataset_dir> <out.npz> [stride]

Each sampled training frame is brightened by a gain (pixel values multiplied, then clipped at
white), a uniform stand-in for what direct sun did to parts of the camera image during the
recorded rollout. The joint state fed to the policy is left untouched, so only the image changes.
"""
import sys
import tempfile

import numpy as np

from act_offline import OfflinePolicy, extract_frames, load_image, recorded_chunk, table

policy_dir, train_dir, out_path = sys.argv[1:4]
STRIDE = int(sys.argv[4]) if len(sys.argv) > 4 else 30  # every 30th frame: 113 frames of the hackathon data
GAINS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

policy = OfflinePolicy(policy_dir)
state, action, episode = table(train_dir)
idx, preds, targets, saturated = [], [], [], []
with tempfile.TemporaryDirectory() as tmp:
    for k, path in enumerate(extract_frames(train_dir, STRIDE, tmp, "train")):
        n = k * STRIDE
        if n >= len(state):
            break
        image = load_image(path)
        per_gain, sat = [], []
        for g in GAINS:
            bright = (image * g).clamp(0, 1)
            per_gain.append(policy.predict(bright, state[n]))
            sat.append(float((bright.mean(0) >= 250 / 255).float().mean()))
        idx.append(n), preds.append(np.stack(per_gain)), saturated.append(sat)
        targets.append(recorded_chunk(action, episode, n, policy.chunk_size))
        print(n, [round(float(np.nanmean(np.abs(p - targets[-1]))), 1) for p in per_gain], flush=True)
np.savez_compressed(
    out_path,
    gains=np.array(GAINS),
    frame=np.array(idx),
    pred=np.stack(preds).astype(np.float32),  # frames x gains x chunk x joints
    recorded=np.stack(targets),
    saturated=np.array(saturated),  # frames x gains: share of pixels at or near white
)
print("wrote", out_path)
