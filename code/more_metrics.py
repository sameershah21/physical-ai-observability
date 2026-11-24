"""Four further metrics, computed from result files and recordings. Prints every number the post quotes.

    uv run --no-project --with numpy --with pandas --with pyarrow --with pillow \
        python more_metrics.py <ball_train_dir> <ball_unseen_dir> <ball_rollout_dir> <toaster_dataset_dir> <results_dir> <out.json>

1. Which input an answer comes from (attribution_*.npz, bread_attribution_n*.npz).
2. Coverage: how far the demonstrations a policy never saw are from the closest stretch of its
   training demonstrations. Needs only the datasets, no policy.
3. The first action of a chunk, and from which position in the chunk a policy beats holding still.
4. How far a picture is from the training pictures in the policy's own vision features.
"""
import glob
import json
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

ball_train, ball_unseen, ball_rollout, toaster_dir, R, out_path = sys.argv[1:7]
H, ARM = 100, slice(0, 6)  # chunk length; the toaster dataset's first six values are the arm that moves
SIZES = {2: "bread_n2_020000", 8: "bread_n8_020000", 29: "bread_n29_020000", 58: "bread_020000"}
out = {}


def actions(d):
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(d + "/data/chunk-*/*.parquet"))])
    return np.stack(df["action"].to_numpy()).astype(np.float32), df["episode_index"].to_numpy()


def stretches(a, ep, episodes, step=1):
    """Every H-step stretch of the given episodes; past the end the last action is repeated, as the dataset does."""
    bank = []
    for e in episodes:
        x = a[ep == e]
        x = np.concatenate([x, np.repeat(x[-1:], H - 1, axis=0)])
        bank += [x[k:k + H] for k in range(0, int((ep == e).sum()), step)]
    return np.stack(bank)


def closest(chunks, bank, block=32):
    d = np.empty(len(chunks), dtype=np.float32)
    for s in range(0, len(chunks), block):
        d[s:s + block] = np.abs(bank[None] - chunks[s:s + block, None]).mean((2, 3)).min(1)
    return d


err = lambda p, r: np.nanmean(np.abs(p - r), axis=(1, 2))

# ------------------------------------------------------------------ 1. which input an answer comes from
print("1. Which input an answer comes from")
rows = []
for label, name in [("ball task, 2 steps (the hackathon policy)", "000002")] + [(f"ball task, {int(k):,} steps", k) for k in ("000300", "001000", "003000", "005000", "010000", "020000")] + [("ball task, 20,000 steps with augmentation", "augmented_020000")]:
    d = np.load(f"{R}/attribution_{name}.npz")
    total, cam, joints = (float(np.abs(d[k]).mean()) for k in ("total", "phi_image", "phi_state"))
    rows.append({"policy": label, "answers_differ_by": total, "camera": cam, "joint_readings": joints})
    print(f"  {label:44s}: answers differ by {total:5.2f} between two frames | camera {cam:5.2f} | joint readings {joints:.3f}")
for n in SIZES:
    d = np.load(f"{R}/bread_attribution_n{n}.npz")
    size = np.abs(d["phi"][..., ARM]).mean((0, 2, 3))
    share = 100 * size / size.sum()
    rows.append({"policy": f"toaster task, {n} demonstrations", "answers_differ_by": float(np.abs(d["total"][..., ARM]).mean()),
                 "front_camera_percent": float(share[0]), "hand_camera_percent": float(share[1]), "joint_readings_percent": float(share[2]), "joint_readings": float(size[2])})
    print(f"  toaster task, {n:2d} demonstrations             : answers differ by {np.abs(d['total'][..., ARM]).mean():5.2f} | front camera {share[0]:.0f}% | hand camera {share[1]:.0f}% | joint readings {share[2]:.1f}% ({size[2]:.2f})")
out["input_shares"] = rows

# ------------------------------------------------------------------ 2. coverage
print("2. Coverage: distance from what was demonstrated in unseen demonstrations to the closest training stretch")
a, ep = actions(toaster_dir)
a = a[:, ARM]
rows = []
for n, f in SIZES.items():
    d = np.load(f"{R}/{f}.npz")
    held = d["held_out"]
    pred, rec, state = d["pred"][held][:, 0][..., ARM], d["recorded"][held][..., ARM], d["state"][held][..., ARM]
    whole = ~np.isnan(rec).any((1, 2))
    bank = stretches(a, ep, range(n), step=2)
    cover = closest(rec[whole], bank)
    near = closest(pred, bank)
    e = err(pred, rec)
    hold = err(np.repeat(state[:, None, :], H, axis=1), rec)
    rows.append({"demonstrations": n, "coverage": float(cover.mean()), "error_on_unseen": float(e.mean()), "holding_still": float(hold.mean()),
                 "prediction_to_closest_stretch": float(near.mean()), "closer_to_a_stretch_than_to_the_truth": int((near < e).sum()), "frames": int(len(e))})
    print(f"  {n:2d} demonstrations: coverage {cover.mean():5.2f} | error on unseen {e.mean():5.2f} | holding still {hold.mean():5.2f}"
          f" | predictions are {near.mean():.2f} from the closest training stretch, closer to it than to the truth in {int((near < e).sum())} of {len(e)}")
out["coverage_toaster"] = rows
ba, bep = actions(ball_train)
ua, uep = actions(ball_unseen)
u = np.load(f"{R}/unseen_demo_020000.npz")
whole = ~np.isnan(u["train_recorded"]).any((1, 2))
cover = closest(u["train_recorded"][whole], stretches(ba, bep, (0, 1)))
out["coverage_ball"] = {"coverage": float(cover.mean()), "error_on_unseen": float(err(u["train_pred"], u["train_recorded"]).mean())}
print(f"  ball task, 2 demonstrations: coverage {cover.mean():.2f} | error on the unseen demonstration {out['coverage_ball']['error_on_unseen']:.2f}")

# ------------------------------------------------------------------ 3. inside a chunk
print("3. Inside a chunk, on demonstrations the policy never saw")
rows = []
for n, f in SIZES.items():
    d = np.load(f"{R}/{f}.npz")
    held = d["held_out"]
    pred, rec, state = d["pred"][held][:, 0][..., ARM], d["recorded"][held][..., ARM], d["state"][held][..., ARM]
    by_k = np.nanmean(np.abs(pred - rec), axis=(0, 2))
    hold_k = np.nanmean(np.abs(state[:, None, :] - rec), axis=(0, 2))
    worse = np.where(by_k >= hold_k)[0]
    cross = int(worse.max()) + 2 if len(worse) else 1
    jump, lead = float(np.abs(pred[:, 0] - state).mean()), float(np.abs(rec[:, 0] - state).mean())
    rows.append({"demonstrations": n, "first_action_from_the_arm": jump, "demonstrators_first_action_from_the_arm": lead, "beats_holding_still_from_step": cross,
                 "error_at_step_1": float(by_k[0]), "holding_still_at_step_1": float(hold_k[0]), "error_at_step_100": float(by_k[-1]), "holding_still_at_step_100": float(hold_k[-1])})
    print(f"  {n:2d} demonstrations: first action is {jump:5.2f} from the arm (the demonstrator's: {lead:.2f}) | better than holding still from step {cross} of {H} ({cross / 30:.1f} s)"
          f" | error at step 1: {by_k[0]:.1f} (holding still {hold_k[0]:.1f}), at step 100: {by_k[-1]:.1f} ({hold_k[-1]:.1f})")
out["inside_a_chunk"] = rows
p = np.load(f"{R}/policy_020000.npz")
out["first_action_ball"] = {
    "own_data": float(np.abs(p["train_pred"][:, 0] - p["train_state"]).mean()), "own_data_demonstrator": float(np.nanmean(np.abs(p["train_recorded"][:, 0] - p["train_state"]))),
    "unseen": float(np.abs(u["train_pred"][:, 0] - u["train_state"]).mean()), "rollout_frames": float(np.abs(p["eval_pred"][:, 0] - p["eval_state"]).mean())}
print("  ball task, 20,000 steps: first action from the arm:", {k: round(v, 1) for k, v in out["first_action_ball"].items()})

# ------------------------------------------------------------------ 4. how far a picture is from the training pictures
print("4. Distance of a picture from the training pictures, in the vision features of the 20,000-step policy")
d = np.load(f"{R}/attribution_020000.npz")
tr, fr = d["train_feature"], d["train_frame"]


def to_train(x, exclude=None):
    dist = np.linalg.norm(tr[None] - x[:, None], axis=2)
    if exclude is not None:
        dist[exclude] = np.inf
    return dist.min(1)


base = to_train(tr, exclude=np.abs(fr[None] - fr[:, None]) <= 150)  # leave out neighbours within 5 seconds
unit = float(np.median(base))


def near_white_table(d_dir, stride):
    video = sorted(glob.glob(d_dir + "/videos/*/chunk-*/*.mp4"))[0]
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vf", f"select='not(mod(n\\,{stride}))'", "-fps_mode", "vfr", f"{tmp}/f_%05d.png"], check=True)
        return np.array([100 * float((np.asarray(Image.open(q).convert("RGB").crop((0, 390, 1130, 1080)), dtype=np.float32).mean(2) >= 250).mean())
                         for q in sorted(glob.glob(f"{tmp}/f_*.png"))])


sun = near_white_table(ball_rollout, int(d["stride"]))[: len(d["rollout_feature"])] > 30
ro = to_train(d["rollout_feature"]) / unit
rows = {"unit": unit, "training_frames_25th_75th": [float(np.percentile(base, q) / unit) for q in (25, 75)], "training_frames_largest": float(base.max() / unit)}
print(f"  unit: the median distance from a training frame to the closest training frame more than 5 s away ({unit:.3f}); middle half of training frames {rows['training_frames_25th_75th'][0]:.2f} to {rows['training_frames_25th_75th'][1]:.2f}, largest {rows['training_frames_largest']:.2f}")
for gi, g in enumerate(d["gains"]):
    v = to_train(d["train_bright_feature"][:, gi]) / unit
    rows[f"brightened_{g:g}"] = float(np.median(v))
    print(f"  training frames brightened by {g:g}: median {np.median(v):.2f}")
un = to_train(d["unseen_feature"]) / unit
rows.update({"unseen_demonstration": float(np.median(un)), "unseen_demonstration_range": [float(un.min()), float(un.max())],
             "rollout_without_sun": float(np.median(ro[~sun])), "rollout_without_sun_range": [float(ro[~sun].min()), float(ro[~sun].max())], "rollout_without_sun_frames": int((~sun).sum()),
             "rollout_with_sun": float(np.median(ro[sun])), "rollout_with_sun_range": [float(ro[sun].min()), float(ro[sun].max())], "rollout_with_sun_frames": int(sun.sum()),
             "rollout_with_sun_beyond_training": int((ro[sun] > rows["training_frames_largest"]).sum())})
print(f"  the unseen demonstration: median {np.median(un):.2f} (from {un.min():.2f} to {un.max():.2f})")
print(f"  rollout frames without sun on the table ({int((~sun).sum())}): median {np.median(ro[~sun]):.2f} (from {ro[~sun].min():.2f} to {ro[~sun].max():.2f})")
print(f"  rollout frames with sun on the table ({int(sun.sum())}): median {np.median(ro[sun]):.2f} (from {ro[sun].min():.2f} to {ro[sun].max():.2f}); beyond the largest training value: {rows['rollout_with_sun_beyond_training']}")
out["picture_distance"] = rows

# ------------------------------------------------------------------ 5. the visual importance ratio of Zhang et al.
print("5. Visual importance ratio R_img (visual_importance_ratio.py), with an input replaced by zeros and by its training mean")
rows = []
for label, name in (("ball task, 2 steps", "ball_000002"), ("ball task, 20,000 steps", "ball_020000")) + tuple((f"toaster task, {n} demonstrations", f"bread_n{n}") for n in SIZES):
    row = {"policy": label}
    for kind, prefix in (("zeros", "rimg_"), ("training_mean", "rimg_mean_")):
        with open(f"{R}/{prefix}{name}.json") as f:
            r = json.load(f)
        row[kind] = {k: r[k] for k in ("E_img", "E_prop", "R_img_percent")}
    rows.append(row)
    print(f"  {label:32s}: zeros {row['zeros']['R_img_percent']:5.1f}% (E_img {row['zeros']['E_img']:.0f}, E_prop {row['zeros']['E_prop']:.1f})"
          f" | training mean {row['training_mean']['R_img_percent']:5.1f}% (E_img {row['training_mean']['E_img']:.0f}, E_prop {row['training_mean']['E_prop']:.1f})")
out["visual_importance_ratio"] = rows

with open(out_path, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", out_path)
