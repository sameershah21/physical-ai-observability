"""Builds every figure of the post.

    uv run --no-project --with matplotlib --with pandas --with pyarrow --with pillow --with numpy \
        python make_figures.py <train_dataset_dir> <eval_dataset_dir> <results_dir> <out_dir>

Schematics (1-4) are drawn from scratch. Figures 5-7 are plotted from the recorded
demonstrations, the recorded policy rollout, and the offline test of the hackathon policy
(<results_dir>/policy_000002.npz). Figures 8-10 need the re-run: policy_<steps>.npz for
each tested checkpoint and lighting_<steps>.npz for the last one; if augmented_lighting_<steps>.npz
is there, the light test also shows the run trained with image augmentation. Figure 11 needs
the unseen-demonstration and 58-demonstration results, figure 12 the runs on 2, 8 and 29
demonstrations (bread_n<N>_020000.npz).
"""
import glob, os, subprocess, sys, tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

train_dir, eval_dir, results_dir, out_dir = sys.argv[1:5]

BLUE, TEAL, RED, GREEN, GRAY, GOLD = "#2f6fb7", "#1a9e96", "#c0392b", "#2e8b57", "#555555", "#b7791f"
PURPLE = "#7b4fb3"  # second series beside the green of the re-run; green and red together are not told apart by every reader
FPS = 30
EXTRA = os.environ.get("EXTRA_FIGURES") == "1"  # also draw the two schematics the compact post leaves out
JOINTS = ["shoulder pan", "shoulder lift", "elbow flex", "wrist flex", "wrist roll", "gripper"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})


def save(fig, name):
    fig.savefig(f"{out_dir}/{name}", dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", name)


def box(ax, xy, w, h, text, color, face, fontsize=10):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", lw=1.6, ec=color, fc=face))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, a, b, color=GRAY, rad=0.0, lw=1.8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=lw, color=color, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------- 1. MDP loop
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.set_xlim(0, 10), ax.set_ylim(0, 5), ax.axis("off")
box(ax, (5, 4.1), 3.6, 1.0, "Policy\n" + r"$\pi_\theta(a_t \mid s_t)$", BLUE, "#e3eefb")
box(ax, (5, 0.9), 3.6, 1.0, "Environment\n" + r"$\mathcal{D}(s_{t+1} \mid s_t, a_t)$", GREEN, "#e4f3ea")
arrow(ax, (6.85, 4.1), (6.85, 0.9), RED, rad=-0.75)
arrow(ax, (3.15, 0.9), (3.15, 4.1), GREEN, rad=-0.75)
ax.text(8.65, 2.5, r"action $a_t$", color=RED, ha="left", va="center")
ax.text(1.35, 2.5, "next state $s_{t+1}$\nreward $r_t$", color=GREEN, ha="right", va="center")
save(fig, "mdp-loop.png") if EXTRA else plt.close(fig)

# ---------------------------------------------------------- 2. covariate shift
x = np.linspace(-4, 7, 600)
g = lambda m: np.exp(-0.5 * (x - m) ** 2) / np.sqrt(2 * np.pi)
fig, ax = plt.subplots(figsize=(6.4, 2.7))
ax.fill_between(x, g(0), color=BLUE, alpha=0.30), ax.plot(x, g(0), color=BLUE, lw=2, label=r"$p_{\mathrm{train}}(o)$")
ax.fill_between(x, g(2.2), color=RED, alpha=0.30), ax.plot(x, g(2.2), color=RED, lw=2, label=r"$p_{\mathrm{deploy}}(o)$")
ax.axvspan(2.2, 5.6, color="#000000", alpha=0.05)
ax.text(3.9, 0.535, "observations the policy\nwas never trained on", ha="center", va="center", fontsize=9, color=RED)
ax.annotate("", xy=(2.2, 0.445), xytext=(0, 0.445), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5))
ax.text(1.1, 0.465, "shift", ha="center", color=GRAY, fontsize=9)
ax.set_ylim(0, 0.62), ax.set_xlim(-4, 7), ax.set_yticks([]), ax.set_xticks([])
ax.set_xlabel("observation $o$ (one-dimensional sketch)"), ax.set_ylabel("density")
ax.legend(frameon=False, loc="upper left")
save(fig, "covariate-shift.png")

# ------------------------------------------------------- 3. ACT as a conditional VAE
fig, ax = plt.subplots(figsize=(9.4, 4.3))
ax.set_xlim(0, 18.8), ax.set_ylim(0, 7.6), ax.axis("off")
ax.text(0.1, 7.25, "Training", fontsize=11, fontweight="bold")
box(ax, (2.3, 5.9), 4.2, 1.25, "demonstrated chunk\n" + r"$a_{t:t+H-1}$" + " + joint state", GOLD, "#fbf1dc", 9)
box(ax, (7.5, 5.9), 4.2, 1.25, "encoder\n" + r"$q_\phi(z \mid o_t, a_{t:t+H-1})$", BLUE, "#e3eefb", 9)
box(ax, (11.3, 5.9), 1.7, 1.25, "style\n" + r"$z$", GRAY, "#eeeeee", 9)
box(ax, (15.6, 5.9), 5.0, 1.25, "decoder (the policy)\n" + r"$\pi_\theta(a_{t:t+H-1} \mid o_t, z)$", GREEN, "#e4f3ea", 9)
arrow(ax, (4.45, 5.9), (5.35, 5.9)), arrow(ax, (9.65, 5.9), (10.4, 5.9)), arrow(ax, (12.2, 5.9), (13.05, 5.9))
box(ax, (15.6, 4.0), 5.0, 0.8, r"$o_t$: camera image + joint state", GRAY, "#ffffff", 9)
arrow(ax, (15.6, 4.42), (15.6, 5.24))
ax.text(6.4, 4.05, r"loss $= \;\|a - \hat a\|_1 \;+\; \beta\,D_{\mathrm{KL}}\!\left[q_\phi \,\|\, \mathcal{N}(0, I)\right]$", ha="center", fontsize=10.5)
ax.plot([0.1, 18.7], [3.2, 3.2], color="#cccccc", lw=1)
ax.text(0.1, 2.7, "Deployment", fontsize=11, fontweight="bold")
box(ax, (11.3, 1.75), 1.7, 1.25, "style\n" + r"$z = 0$", GRAY, "#eeeeee", 9)
box(ax, (15.6, 1.75), 5.0, 1.25, "decoder (the policy)\n" + r"$\pi_\theta(a_{t:t+H-1} \mid o_t, z)$", GREEN, "#e4f3ea", 9)
arrow(ax, (12.2, 1.75), (13.05, 1.75))
box(ax, (15.6, 0.35), 5.0, 0.6, r"observation $o_t$", GRAY, "#ffffff", 9)
arrow(ax, (15.6, 0.67), (15.6, 1.09))
ax.text(5.3, 1.75, "The encoder is not used.\nThe decoder outputs a chunk of $H$ actions,\nwhich the robot then executes.", ha="center", va="center", fontsize=9.5, color=GRAY)
save(fig, "act-cvae.png")

# ----------------------------------------------------- 4. diffusion vs flow matching
rng = np.random.default_rng(7)
start, goal = np.array([0.0, 0.0]), np.array([4.0, 2.0])
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
n = 50
t = np.linspace(0, 1, n + 1)[:, None]
noise = np.cumsum(rng.normal(0, 0.16, (n + 1, 2)), axis=0)
noise -= t * noise[-1]  # pinned at both ends
path = start + t * (goal - start) + noise
axes[0].plot(path[:, 0], path[:, 1], "-o", color=RED, lw=1, ms=2.6, alpha=0.9)
axes[0].set_title(f"Diffusion (DDPM sampler)\n{n} small noisy steps shown", fontsize=10)
axes[0].text(*(start + [0.0, -0.45]), r"noise $z_T$", ha="center", fontsize=9), axes[0].text(*(goal + [0.3, -0.05]), r"sample $z_0$", ha="left", va="center", fontsize=9)
m = 10
line = start + np.linspace(0, 1, m + 1)[:, None] * (goal - start)
axes[1].plot(line[:, 0], line[:, 1], "-o", color=GREEN, lw=1.6, ms=4.5)
axes[1].set_title(f"Flow matching\n{m} steps along the learned velocity field", fontsize=10)
axes[1].text(*(start + [0.0, -0.45]), r"noise $z_0$", ha="center", fontsize=9), axes[1].text(*(goal + [0.3, -0.05]), r"sample $z_1$", ha="left", va="center", fontsize=9)
for a in axes:
    a.scatter(*start, s=45, color=GRAY, zorder=5), a.scatter(*goal, s=45, color="black", zorder=5)
    a.set_xlim(-1.0, 5.9), a.set_ylim(-1.1, 3.1), a.axis("off")
save(fig, "diffusion-vs-flow.png") if EXTRA else plt.close(fig)


# --------------------------------------------------------------- recorded data
def table(d):
    df = pd.read_parquet(sorted(glob.glob(d + "/data/chunk-*/*.parquet"))[0])
    stack = lambda c: np.stack(df[c].to_numpy())
    return stack("observation.state"), stack("action"), df["episode_index"].to_numpy()


tr_s, tr_a, tr_ep = table(train_dir)
ev_s, ev_a, _ = table(eval_dir)


def per_frame_error(pred, recorded):
    """Mean absolute difference per sampled frame, over the steps that were recorded and all joints."""
    return np.nanmean(np.abs(pred - recorded), axis=(1, 2))


def travel(chunks):
    """How far each joint moves inside a chunk (max minus min), averaged over frames and joints."""
    return float(np.nanmean(np.nanmax(chunks, 1) - np.nanmin(chunks, 1)))


hack = np.load(f"{results_dir}/policy_000002.npz")
H = int(hack["chunk_size"])

# -------------------------------------------- 5. demonstrations vs recorded rollout
fig = plt.figure(figsize=(7.6, 6.2))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.05], hspace=0.62, wspace=0.32)
for j, name in enumerate(JOINTS):
    ax = fig.add_subplot(gs[j // 3, j % 3])
    for ep, color in ((0, BLUE), (1, TEAL)):
        s = tr_s[tr_ep == ep][:, j]
        ax.plot(np.arange(len(s)) / FPS, s, color=color, lw=1.2, label=f"demonstration {ep + 1}")
    ax.plot(np.arange(len(ev_s)) / FPS, ev_s[:, j], color=RED, lw=1.8, label="policy rollout")
    ax.set_title(name, fontsize=10)
    if j >= 3:
        ax.set_xlabel("time (s)")
    if j % 3 == 0:
        ax.set_ylabel("joint position")
    if j == 0:
        handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.965))
ax = fig.add_subplot(gs[2, :])
j = 1
steps = np.arange(len(ev_a))
ax.plot(steps, ev_a[:, j], color=RED, lw=1.3)
for k in range(0, len(ev_a) + 1, H):
    ax.axvline(k, color="#999999", lw=0.8, ls="--")
ax.set_xlim(0, len(ev_a)), ax.set_xlabel(f"control step of the rollout (dashed line every {H} steps)"), ax.set_ylabel("commanded\nshoulder lift")
ax.set_title("Shoulder-lift command sent to the arm during the rollout", fontsize=10)
save(fig, "rollout-vs-demos.png")
step_change = np.abs(np.diff(ev_a, axis=0))
at_boundary = np.zeros(len(step_change), bool)
at_boundary[np.arange(H, len(ev_a), H) - 1] = True
print(f"  rollout commands: mean change across a chunk boundary {step_change[at_boundary].mean():.2f} | between other steps {step_change[~at_boundary].mean():.2f}")
print("  step of each joint's largest command change:", (step_change.argmax(0) + 1).tolist())

# ------------------------------------------------------ 6. lighting, same camera
CROP = (0, 390, 1130, 1080)  # left, top, right, bottom: table area only


def frame(d, n):
    video = sorted(glob.glob(d + "/videos/*/chunk-*/*.mp4"))[0]
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vf", f"select='eq(n\\,{n})'", "-fps_mode", "vfr", "-frames:v", "1", f"{tmp}/f.png"], check=True)
        return Image.open(f"{tmp}/f.png").convert("RGB").crop(CROP)


def near_white(im):
    """Share of pixels whose three colour values average at least 250 of 255, in percent."""
    return 100 * float((np.asarray(im.convert("RGB"), dtype=np.float32).mean(2) >= 250).mean())


def white_shares(d, stride):
    """near_white of every stride-th frame of a recording: over the whole picture, and over the table area."""
    video = sorted(glob.glob(d + "/videos/*/chunk-*/*.mp4"))[0]
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vf", f"select='not(mod(n\\,{stride}))'", "-fps_mode", "vfr", f"{tmp}/f_%05d.png"], check=True)
        frames_ = [Image.open(path) for path in sorted(glob.glob(f"{tmp}/f_*.png"))]
        return np.array([near_white(im) for im in frames_]), np.array([near_white(im.crop(CROP)) for im in frames_])


first_ep1 = int(np.argmax(tr_ep == 1))
left, right = frame(train_dir, first_ep1 + 20), frame(eval_dir, 1500)
fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.75))
for a, im, title in ((axes[0], left, "During the demonstrations"), (axes[1], right, "During the policy rollout, about two hours later")):
    a.imshow(im), a.set_title(title, fontsize=10), a.axis("off")
fig.subplots_adjust(wspace=0.03)
save(fig, "lighting.png")
print(f"  the two frames shown, table area: pixels at or near white {near_white(left):.1f}% (demonstrations) and {near_white(right):.1f}% (rollout, step 1500)")
demo_whole, demo_table = white_shares(train_dir, 30)
roll_whole, roll_table = white_shares(eval_dir, 100)
for tag, whole, area in (("demonstrations", demo_whole, demo_table), ("rollout", roll_whole, roll_table)):
    print(f"  {tag}, {len(whole)} frames: whole picture {whole.mean():.1f}% near white (from {whole.min():.1f} to {whole.max():.1f}) | table area {area.mean():.1f}% (from {area.min():.1f} to {area.max():.1f})")
print("  rollout, table area by sampled frame:", np.round(roll_table, 0).astype(int).tolist(), "| frames above 30%:", int((roll_table > 30).sum()), "of", len(roll_table))

# ------------------------------------- 7. offline: policy output on its own training data
show = [1, 2, 5]
t_all = np.arange(len(tr_a)) / FPS
boundary = first_ep1 / FPS
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5), sharex=True)
frames_, means = hack["train_frame"], hack["train_pred"].mean(1)
for a, j in zip(axes, show):
    a.plot(t_all, tr_a[:, j], color=GRAY, lw=1.1, label="demonstrated action")
    a.axhline(np.median(tr_a[:, j]), color=BLUE, lw=1.1, ls="--", label="median of the demonstrations")
    a.plot(frames_ / FPS, means[:, j], "o", color=RED, ms=3.2, label="policy prediction")
    a.axvline(boundary, color="#bbbbbb", lw=0.8, ls=":")
    a.set_title(JOINTS[j], fontsize=10), a.set_xlabel("time in the training data (s)")
axes[0].set_ylabel("joint position")
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.09))
fig.subplots_adjust(wspace=0.28)
save(fig, "policy-output.png")

hold = np.repeat(hack["train_state"][:, None, :], H, axis=1)
print("  hackathon policy, std of prediction across training frames:", np.round(means.std(0), 2).tolist())
print("  demonstrated action, std:", np.round(tr_a.std(0), 1).tolist())
print("  hackathon policy, mean prediction:", np.round(means.mean(0), 1).tolist(), "| median of demonstrations:", np.round(np.median(tr_a, 0), 1).tolist())
print("  rollout frames: offline prediction vs command sent, mean abs diff:", round(float(per_frame_error(hack["eval_pred"], hack["eval_recorded"]).mean()), 2))
print("  training frames: error of hackathon policy", round(float(per_frame_error(hack["train_pred"], hack["train_recorded"]).mean()), 1),
      "| error of holding still", round(float(per_frame_error(hold, hack["train_recorded"]).mean()), 1))
print("  rollout range of motion per joint:", np.round(ev_s.max(0) - ev_s.min(0), 1).tolist(), "| demonstrations:", np.round(tr_s.max(0) - tr_s.min(0), 1).tolist())

# ------------------------------------------------ the re-run: needs the tested checkpoints
runs = {}
for path in sorted(glob.glob(f"{results_dir}/policy_*.npz")):
    runs[int(os.path.basename(path)[7:-4])] = np.load(path)
steps = sorted(runs)
if len(steps) < 3:
    sys.exit("re-run figures skipped: fewer than three tested checkpoints")
last = runs[steps[-1]]
GREEN2 = "#1f8a4c"

# ------------------------------------------------------------------ 8. learning curve
err = [float(per_frame_error(runs[k]["train_pred"], runs[k]["train_recorded"]).mean()) for k in steps]
mov = [travel(runs[k]["train_pred"]) for k in steps]
err_hold = float(per_frame_error(hold, hack["train_recorded"]).mean())
mov_demo = travel(hack["train_recorded"])
fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
for a, values, ref, ref_label, title in (
    (axes[0], err, err_hold, "holding still", "Error on its own training data"),
    (axes[1], mov, mov_demo, "the demonstrations", "Movement inside a predicted chunk"),
):
    a.axhline(ref, color=GRAY, lw=1.1, ls="--")
    a.text(steps[0], ref, ref_label, color=GRAY, fontsize=8, ha="left", va="bottom")
    a.plot(steps, values, "-o", color=GREEN2, ms=4.5, lw=1.5)
    a.plot(steps[0], values[0], "o", color=RED, ms=6.5)
    a.annotate("the failed run\n(2 steps)", (steps[0], values[0]), xytext=(9, -3), textcoords="offset points", color=RED, fontsize=8, va="center")
    a.set_xscale("log"), a.set_xlabel("optimisation steps"), a.set_title(title, fontsize=10), a.set_ylim(bottom=0)
    a.set_xticks([2, 10, 100, 1000, 10000]), a.set_xticklabels(["2", "10", "100", "1,000", "10,000"]), a.minorticks_off()
axes[0].set_ylabel("joint units")
fig.subplots_adjust(wspace=0.25)
save(fig, "learning-curve.png")
for k, e, m in zip(steps, err, mov):
    print(f"  steps {k:6d}: error {e:5.1f} | movement inside a chunk {m:5.1f}")
print(f"  references: holding still {err_hold:.1f} | demonstrated movement over {H} steps {mov_demo:.1f}")

# ------------------------------------------- 9. the trained policy on its training data
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5), sharex=True)
for a, j in zip(axes, show):
    a.plot(t_all, tr_a[:, j], color=GRAY, lw=1.1, label="demonstrated action")
    for n, chunk, rec in zip(last["train_frame"], last["train_pred"], last["train_recorded"]):
        valid = int(np.sum(~np.isnan(rec[:, 0])))
        a.plot((n + np.arange(valid)) / FPS, chunk[:valid, j], color=GREEN2, lw=1.0, alpha=0.75,
               label="predicted chunk" if n == last["train_frame"][0] else None)
    a.axvline(boundary, color="#bbbbbb", lw=0.8, ls=":")
    a.set_title(JOINTS[j], fontsize=10), a.set_xlabel("time in the training data (s)")
axes[0].set_ylabel("joint position")
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.09))
fig.subplots_adjust(wspace=0.28)
save(fig, "trained-policy.png")

# ------------------------------------------------------------------- 10. the light test
light_path = f"{results_dir}/lighting_{steps[-1]:06d}.npz"
if not os.path.exists(light_path):
    sys.exit("light-test figure skipped: no lighting result for the last checkpoint")
light = np.load(light_path)
sat = 100 * light["saturated"].mean(0)
light_err = [float(per_frame_error(light["pred"][:, i], light["recorded"]).mean()) for i in range(len(light["gains"]))]
rest = hack["eval_state"].mean(0)
near = np.abs(last["train_state"] - rest).mean(1) < 6
in_demo_light = last["train_pred"][near].mean((0, 1))
in_sun = last["eval_pred"].mean((0, 1))
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw={"width_ratios": [1, 1.15]})
a = axes[0]
a.plot(sat, light_err, "-o", color=GREEN2, ms=4.5, lw=1.5, label="default training")
aug_path = f"{results_dir}/augmented_lighting_{steps[-1]:06d}.npz"
if os.path.exists(aug_path):
    aug = np.load(aug_path)
    aug_err = [float(per_frame_error(aug["pred"][:, i], aug["recorded"]).mean()) for i in range(len(aug["gains"]))]
    a.plot(100 * aug["saturated"].mean(0), aug_err, "-s", color=PURPLE, ms=4.5, lw=1.5, label="with image augmentation")
    a.legend(frameon=False, fontsize=8, loc="upper left", borderaxespad=0.2)
a.axvspan(roll_whole.min(), roll_whole.max(), color=RED, alpha=0.13, lw=0)
a.axvline(roll_whole.mean(), color=RED, lw=1.0, ls=":")
a.set_xlabel("pixels at or near white, whole picture (%)"), a.set_ylabel("error against the demonstrated actions")
a.set_title("Training frames, brightened", fontsize=10), a.set_ylim(bottom=0)
a.text(roll_whole.max() + 1.2, 0.5, "the recorded rollout:\nits range and its average", color=RED, fontsize=8, va="bottom", ha="left")
a = axes[1]
y = np.arange(len(JOINTS))[::-1]
a.hlines(y, np.minimum(in_demo_light, in_sun), np.maximum(in_demo_light, in_sun), color="#cccccc", lw=2)
a.plot(rest, y, "|", color="black", ms=13, mew=1.6, label="where the arm is")
a.plot(in_demo_light, y, "o", color=BLUE, ms=6, label="predicted, demonstration light")
a.plot(in_sun, y, "s", color=RED, ms=5.5, label="predicted, rollout light")
a.set_yticks(y), a.set_yticklabels(JOINTS), a.set_xlabel("joint position")
a.set_title("Same resting pose, two lights", fontsize=10)
fig.legend(*a.get_legend_handles_labels(), frameon=False, fontsize=8.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
fig.subplots_adjust(wspace=0.42, bottom=0.27)
save(fig, "light-test.png")
for g, st_, e in zip(light["gains"], sat, light_err):
    print(f"  gain {g:4.2f}: {st_:5.1f}% near white | error {e:5.1f}")
print("  resting pose:", np.round(rest, 1).tolist())
print(f"  predicted in demonstration light ({int(near.sum())} frames):", np.round(in_demo_light, 1).tolist())
print(f"  predicted in rollout light ({len(last['eval_pred'])} frames):", np.round(in_sun, 1).tolist())
print("  distance from the resting pose: demonstration light", round(float(np.abs(in_demo_light - rest).mean()), 1), "| rollout light", round(float(np.abs(in_sun - rest).mean()), 1))
if os.path.exists(aug_path):
    for g, st_, e, e0 in zip(aug["gains"], 100 * aug["saturated"].mean(0), aug_err, light_err):
        print(f"  with augmentation, gain {g:4.2f}: {st_:5.1f}% near white | error {e:5.2f} (default training {e0:5.2f})")
    # the same policy on its training data, on the sunlit rollout frames and on the unseen demonstration
    ap = np.load(f"{results_dir}/augmented_policy_{steps[-1]:06d}.npz")
    print(f"  with augmentation: error on its training data {per_frame_error(ap['train_pred'], ap['train_recorded']).mean():.2f} (default {per_frame_error(last['train_pred'], last['train_recorded']).mean():.2f})"
          f" | movement inside a chunk {travel(ap['train_pred']):.1f} (default {travel(last['train_pred']):.1f})")
    near_a = np.abs(ap["train_state"] - rest).mean(1) < 6
    print(f"  with augmentation: distance from the resting pose, demonstration light {np.abs(ap['train_pred'][near_a].mean((0, 1)) - rest).mean():.1f} | rollout frames {np.abs(ap['eval_pred'].mean((0, 1)) - rest).mean():.1f}")
    sunny = roll_table > 30  # rollout frames with the sun on the table area
    for name, d in (("default training", last), ("with augmentation", ap)):
        per = np.abs(d["eval_pred"].mean(1) - rest).mean(1)
        print(f"  {name}: distance from rest on the {int(sunny.sum())} rollout frames with sun on the table area {per[sunny].mean():.1f} | on the other {int((~sunny).sum())} {per[~sunny].mean():.1f}")
    au_path = f"{results_dir}/augmented_unseen_demo_{steps[-1]:06d}.npz"
    if os.path.exists(au_path):
        au = np.load(au_path)
        print(f"  with augmentation: error on the unseen demonstration {per_frame_error(au['train_pred'], au['train_recorded']).mean():.2f} over {len(au['train_frame'])} frames")

# ------------------------------------------- 11. demonstrations seen and not seen
need = [f"{results_dir}/unseen_demo_000002.npz", f"{results_dir}/unseen_demo_020000.npz"]
if not all(os.path.exists(p) for p in need):
    sys.exit("seen-vs-unseen figure skipped: missing unseen-demonstration results")
u2, u20 = (np.load(p) for p in need)
hold_of = lambda state, rec: float(per_frame_error(np.repeat(state[:, None, :], rec.shape[1], axis=1), rec).mean())
groups = [
    ("2 demonstrations\n2 steps\n(the hackathon policy)",
     float(per_frame_error(hack["train_pred"], hack["train_recorded"]).mean()), float(per_frame_error(u2["train_pred"], u2["train_recorded"]).mean()),
     hold_of(u2["train_state"], u2["train_recorded"])),
    ("2 demonstrations\n20,000 steps",
     float(per_frame_error(last["train_pred"], last["train_recorded"]).mean()), float(per_frame_error(u20["train_pred"], u20["train_recorded"]).mean()),
     hold_of(u20["train_state"], u20["train_recorded"])),
]
fig, ax = plt.subplots(figsize=(5.4, 3.5))
x = np.arange(len(groups))
seen_vals, unseen_vals, holds = [g[1] for g in groups], [g[2] for g in groups], [g[3] for g in groups]
BAR = 0.2
b1 = ax.bar(x - BAR / 2 - 0.012, seen_vals, BAR, color=BLUE, label="the demonstrations it trained on")
b2 = ax.bar(x + BAR / 2 + 0.012, unseen_vals, BAR, color=RED, label="the demonstration it never saw")
for xi, h in zip(x, holds):
    ax.plot([xi + 0.012, xi + 0.012 + BAR], [h, h], color="black", lw=1.6, ls="--", label="holding still, on the unseen one" if xi == 0 else None)
for bars in (b1, b2):
    for rect in bars:
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.8, f"{rect.get_height():.1f}", ha="center", fontsize=8.5)
ax.set_xticks(x), ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
ax.set_ylabel("error against the demonstrated actions"), ax.set_ylim(0, max(unseen_vals) * 1.12), ax.set_xlim(-0.55, len(groups) - 0.45)
ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, "seen-vs-unseen.png")
for g in groups:
    print(f"  {g[0].splitlines()[0]} / {g[0].splitlines()[1]}: seen {g[1]:.1f} | unseen {g[2]:.1f} | holding still on the unseen {g[3]:.1f}")


# ------------------------------- 13. where the two-demonstration policy's answers come from
# Every 100-step stretch of the two training demonstrations, compared with each chunk the trained
# policy predicts: on its training frames as recorded and brightened (the light test), on the
# demonstration it never saw, and on the frames of the sunlit rollout. Stretch i starts at
# frame i of the training data.
stretches = []
for ep in (0, 1):
    a_ep = tr_a[tr_ep == ep]
    a_ep = np.concatenate([a_ep, np.repeat(a_ep[-1:], H - 1, axis=0)])  # past the end the dataset repeats the last action
    stretches += [a_ep[k:k + H] for k in range(int((tr_ep == ep).sum()))]
stretches = np.stack(stretches).astype(np.float32)
TOL = 1.0  # stretches this close to the closest one count as equally close: the arm waits in places


def closest_stretch(pred, own_frame=None):
    """Distance to the closest stretch, where it starts, and the distance to the stretch at the frame's own place."""
    dist, start, own = [], [], []
    for i, chunk in enumerate(pred):
        d = np.abs(stretches - chunk[None]).mean((1, 2))
        k = int(d.argmin())
        dist.append(float(d[k]))
        if own_frame is not None:
            close = np.where(d <= d[k] + TOL)[0]
            k = int(close[np.abs(close - own_frame[i]).argmin()])
            own.append(float(d[own_frame[i]]))
        start.append(k)
    return np.array(dist), np.array(start), np.array(own)


fr = light["frame"]
placed = {}
for i, g in enumerate(light["gains"]):
    dist, start, own = closest_stretch(light["pred"][:, i], fr)
    lost = (tr_ep[start] != tr_ep[fr]) | (np.abs(start - fr) > H)
    placed[float(g)] = start
    print(f"  training frames, gain {g:4.2f} ({sat[i]:.1f}% near white): error {light_err[i]:.2f} | closest stretch {dist.mean():.2f} away"
          f" | own place among the closest: {int((own <= dist + TOL).sum())} of {len(fr)} | closest stretch more than {H} steps from its own place: {int(lost.sum())} of {len(fr)}")
u_dist, u_start, _ = closest_stretch(u20["train_pred"])
u_err = per_frame_error(u20["train_pred"], u20["train_recorded"])
whole = ~np.isnan(u20["train_recorded"]).any((1, 2))
truth_dist, _, _ = closest_stretch(u20["train_recorded"][whole])
two = lambda st: (tr_ep[st] == 0) & (((st >= 400) & (st < 600)) | ((st >= 700) & (st < 800)))
print(f"  unseen demonstration: {len(u_err)} frames | error {u_err.mean():.2f} | closest stretch {u_dist.mean():.2f} away | closer to a stretch than to the truth: {int((u_dist < u_err).sum())} of {len(u_err)}"
      f" | what was demonstrated there is {truth_dist.mean():.2f} from its closest stretch | closest stretch starts in steps 400-599 or 700-799 of demonstration 1: {int(two(u_start).sum())} of {len(u_err)}")
print(f"  unseen demonstration: movement inside a chunk, predicted {travel(u20['train_pred']):.1f} | demonstrated {travel(u20['train_recorded']):.1f}")
s_dist, s_start, _ = closest_stretch(last["eval_pred"])
print(f"  sunlit rollout frames: closest stretch {s_dist.mean():.2f} away | in the same two stretches: {int(two(s_start).sum())} of {len(s_start)}")

split = first_ep1 / FPS
total = len(tr_a) / FPS
fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.05), sharey=True, gridspec_kw={"width_ratios": [1, 1, 0.62]})
for a, gain, color, title in zip(axes, (1.0, 2.0), (BLUE, RED), ("Training frames, as recorded", "The same frames, twice as bright")):
    a.plot([0, total], [0, total], color="#bbbbbb", lw=1.0, zorder=1)
    a.plot(fr / FPS, placed[gain] / FPS, "o", color=color, ms=3.6, mec="white", mew=0.4, zorder=3)
    a.axvline(split, color="#bbbbbb", lw=0.8, ls=":"), a.set_xlim(0, total), a.set_xticks([0, 30, 60, 90])
    a.set_title(title, fontsize=10), a.set_xlabel("where the frame really is (s)")
a = axes[2]
a.plot(u20["train_frame"] / FPS, u_start / FPS, "o", color=RED, ms=3.6, mec="white", mew=0.4, zorder=3)
a.set_title("A demonstration\nit never saw", fontsize=10), a.set_xlabel("time in that\ndemonstration (s)"), a.set_xticks([0, 30, 60])
for a in axes:
    a.axhline(split, color="#bbbbbb", lw=0.8, ls=":"), a.set_ylim(0, total)
axes[0].set_ylabel("closest stretch of the two\ntraining demonstrations starts at (s)")
axes[0].text(2, split - 3, "demonstration 1", fontsize=7.5, color=GRAY, va="top"), axes[0].text(2, total - 3, "demonstration 2", fontsize=7.5, color=GRAY, va="top")
fig.subplots_adjust(wspace=0.1)
save(fig, "where-answers-come-from.png")


# ------------------------------------------ 12. the toaster task: how many demonstrations
# The same settings trained on 2, 8, 29 and 58 demonstrations, each tested on the same six
# demonstrations (episodes 58-63) that none of the runs trained on.
ARM = slice(0, 6)  # the dataset holds two arms; in this task only the first one moves


def split_errors(d):
    """Per brightness gain: error on the trained-on frames, on the unseen frames, and the share of pixels near white on the unseen frames."""
    held = d["held_out"]
    err = np.stack([per_frame_error(d["pred"][:, g][..., ARM], d["recorded"][..., ARM]) for g in range(len(d["gains"]))])
    return err[:, ~held].mean(1), err[:, held].mean(1), 100 * d["saturated"][held].mean(0)


bread_runs = {int(os.path.basename(path)[6:-4]): np.load(path) for path in sorted(glob.glob(f"{results_dir}/bread_0*.npz"))}
if 20000 not in bread_runs:
    sys.exit("toaster figures skipped: no 58-demonstration result")
bread = bread_runs[20000]
bh = bread["held_out"]
hold_unseen = hold_of(bread["state"][bh][..., ARM], bread["recorded"][bh][..., ARM])
hold_seen = hold_of(bread["state"][~bh][..., ARM], bread["recorded"][~bh][..., ARM])
print(f"  toaster task, the arm that moves: holding still scores {hold_unseen:.2f} on the unseen demonstrations and {hold_seen:.2f} on the trained-on ones")
print(f"  toaster task: movement inside a chunk, demonstrated {travel(bread['recorded'][bh][..., ARM]):.1f} | predicted on unseen {travel(bread['pred'][bh][:, 0][..., ARM]):.1f}")
print(f"  toaster task: the second arm's error on unseen demonstrations {per_frame_error(bread['pred'][bh][:, 0][..., 6:], bread['recorded'][bh][..., 6:]).mean():.2f}; its movement inside a chunk {travel(bread['recorded'][..., 6:]):.2f}")
for k, d in bread_runs.items():
    s_, u, w = split_errors(d)
    print(f"  58 demonstrations, {k:6d} steps: trained-on {s_[0]:.2f} | unseen {u[0]:.2f} | by share of pixels near white on the unseen frames, trained-on / unseen:", ", ".join(f"{wi:.1f}%: {si:.2f} / {ui:.2f}" for wi, si, ui in zip(w, s_, u)))

# ------------------------------------------------ 14. the order of questions, with the values found
# A schematic, but every number in it is read from the result files above. The hackathon run's
# own step count (2, against the tool's default of 20,000) is the one value typed in by hand.
g2 = int(np.argmin(np.abs(light["gains"] - 2.0)))
b_seen, b_unseen, _ = split_errors(bread)
bg2 = int(np.argmin(np.abs(bread["gains"] - 2.0)))
i300 = steps.index(300) if 300 in steps else 1
u20_err = float(per_frame_error(u20["train_pred"], u20["train_recorded"]).mean())
u20_hold = hold_of(u20["train_state"], u20["train_recorded"])
theta_line = []
if os.path.exists(f"{results_dir}/theta_movement.json"):
    import json

    with open(f"{results_dir}/theta_movement.json") as f:
        theta = json.load(f)
    first = min(theta["checkpoints"], key=lambda c: c["steps"])
    theta_line = [("", f"no number in $\\theta$ had moved by more than {first['largest_change']:.6f}")]
    print(f"  theta: {theta['values']:,} values, typical size {theta['mean_size_of_a_value']:.4f}, learning rate {theta['learning_rate']:g}")
    for c in theta["checkpoints"]:
        print(f"  theta after {c['steps']:6d} steps: mean change {c['mean_change']:.2e} ({100 * c['mean_change'] / theta['mean_size_of_a_value']:.2f}% of the typical size) | largest {c['largest_change']:.2e}")
joint_share = []
for name in ("attribution_000002", "attribution_020000"):
    if os.path.exists(f"{results_dir}/{name}.npz"):
        z = np.load(f"{results_dir}/{name}.npz")
        cam, joints = float(np.abs(z["phi_image"]).mean()), float(np.abs(z["phi_state"]).mean())
        joint_share.append(100 * joints / (cam + joints))
for path in sorted(glob.glob(f"{results_dir}/bread_attribution_n*.npz")):
    size = np.abs(np.load(path)["phi"][..., ARM]).mean((0, 2, 3))
    joint_share.append(100 * float(size[2] / size.sum()))
shares_line = [("", f"joint readings: {min(joint_share):.0f} to {max(joint_share):.0f}% of an answer; the cameras decide")] if joint_share else []
if joint_share:
    print("  share of an answer that comes from the joint readings (%):", np.round(joint_share, 2).tolist())
toaster_two = []
if os.path.exists(f"{results_dir}/bread_n2_020000.npz"):
    two_demos = np.load(f"{results_dir}/bread_n2_020000.npz")
    two_hold = hold_of(two_demos["state"][two_demos["held_out"]][..., ARM], two_demos["recorded"][two_demos["held_out"]][..., ARM])
    toaster_two = [("fails", f"toaster task, trained on 2: {split_errors(two_demos)[1][0]:.1f}, holding still scores {two_hold:.1f}")]
rows = [
    ("1   Was it trained?", r"each step moves $\theta$ by about $\eta$," + "\nso count the steps",
     [("fails", "the saved training configuration says steps: 2")] + theta_line + [("", "the tool offers 20,000 steps by default")],
     "Train."),
    ("2   Does it fit its data?", r"the loss $\mathcal{L}(\theta)$ on" + "\n" + r"the demonstrations $\mathcal{D}$",
     [("fails", f"after 2 steps: error {err[0]:.1f}, where holding still scores {err_hold:.1f}"),
      ("fails", f"after {steps[i300]} steps: {err[i300]:.1f}, but its chunks hardly move"),
      ("passes", f"after 20,000 steps: {err[-1]:.1f}, and its chunks move as demonstrated")],
     "Train longer. The data and the light can wait."),
    ("3   Does it carry over to a\n     demonstration it never saw?", r"$\mathcal{D}$ holds only $N$ demonstrations",
     [("fails", f"ball task, trained on 2: error {u20_err:.1f}, holding still scores {u20_hold:.1f}")] + toaster_two +
     [("passes", f"toaster task, trained on 58: {b_unseen[0]:.1f}, holding still scores {hold_unseen:.1f}")],
     "Record more demonstrations. More steps will not help."),
    ("4   What does it depend on?", r"$\pi_\theta$ reads the observation $o_t$, and" + "\n" + r"$p_{\mathrm{deploy}}(o_t) \neq p_{\mathrm{train}}(o_t)$",
     shares_line +
     [("fails", f"ball task, picture twice as bright: error {light_err[0]:.1f} becomes {light_err[g2]:.1f}"),
      ("passes", f"toaster task, trained on 58, the same: {b_unseen[0]:.1f} becomes {b_unseen[bg2]:.1f}")],
     "Control the scene, or put its variation in the data."),
]
FW, FH = 7.6, 5.95  # drawn in inches
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW), ax.set_ylim(0, FH), ax.axis("off")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
AL, AW, BL, BW, RH = 0.06, 2.5, 2.86, 4.68, 1.06
ys = [4.86 - 1.28 * i for i in range(4)]
ax.text(AL + AW / 2, FH - 0.3, "The check, and the math behind it", ha="center", va="center", fontsize=9, color=GRAY)
ax.text(BL + BW / 2, FH - 0.3, "What we found, and what to do if it fails", ha="center", va="center", fontsize=9, color=GRAY)
for y, (question, math, found, action) in zip(ys, rows):
    box(ax, (AL + AW / 2, y), AW, RH, "", BLUE, "#e3eefb")
    ax.text(AL + 0.13, y + RH / 2 - 0.13, question, ha="left", va="top", fontsize=8.7, fontweight="bold", linespacing=1.3)
    ax.text(AL + 0.13, y - RH / 2 + 0.11, math, ha="left", va="bottom", fontsize=8.0, color="#333333", linespacing=1.35)
    box(ax, (BL + BW / 2, y), BW, RH, "", "#999999", "white")
    for i, (tag, line) in enumerate(found):
        yy = y + RH / 2 - 0.17 - 0.18 * i
        ax.text(BL + 0.13, yy, tag, ha="left", va="center", fontsize=7.8, fontweight="bold", color=RED if tag == "fails" else BLUE)
        ax.text(BL + 0.7, yy, line, ha="left", va="center", fontsize=8.2)
    ax.add_patch(FancyBboxPatch((BL + 0.09, y - RH / 2 + 0.08), BW - 0.18, 0.25, boxstyle="round,pad=0.0,rounding_size=0.04", lw=0, fc="#fbf1dc"))
    ax.text(BL + 0.17, y - RH / 2 + 0.205, "If it fails:", ha="left", va="center", fontsize=8.0, fontweight="bold")
    ax.text(BL + 0.92, y - RH / 2 + 0.205, action, ha="left", va="center", fontsize=8.2)
    arrow(ax, (AL + AW + 0.05, y), (BL - 0.05, y), GRAY, lw=1.3)
for y0, y1 in zip(ys, ys[1:] + [ys[-1] - 1.28]):
    arrow(ax, (AL + 0.5, y0 - RH / 2 - 0.03), (AL + 0.5, y1 + RH / 2 + 0.03), BLUE, lw=1.5)
    ax.text(AL + 0.62, y0 - RH / 2 - 0.11, "if it passes", ha="left", va="center", fontsize=7.8, color=GRAY)
ax.text(AL + 0.02, ys[-1] - 1.28 + RH / 2 - 0.02, "Only now suspect what a test without a robot cannot see: the hardware, the timing,\nand errors that build up while the policy drives itself.",
        ha="left", va="top", fontsize=8.7, linespacing=1.3)
save(fig, "order-of-questions.png")

SIZES = (2, 8, 29)
if not all(os.path.exists(f"{results_dir}/bread_n{n}_020000.npz") for n in SIZES):
    sys.exit("more-demonstrations figure skipped: the runs on 2, 8 and 29 demonstrations are not all tested yet")
by_n = {n: np.load(f"{results_dir}/bread_n{n}_020000.npz") for n in SIZES}
by_n[58] = bread
ns = sorted(by_n)
seen_n, unseen_n, white_n = zip(*(split_errors(by_n[n]) for n in ns))
# The runs sample slightly different frames of the same six unseen demonstrations, so holding still scores slightly differently in each.
holds_n = [hold_of(by_n[n]["state"][by_n[n]["held_out"]][..., ARM], by_n[n]["recorded"][by_n[n]["held_out"]][..., ARM]) for n in ns]
hold_line = float(np.mean(holds_n))
RAMP = ["#e9a093", "#d86a59", RED, "#8c2419"]  # one hue, darker with more demonstrations
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
a = axes[0]
a.axhline(hold_line, color=GRAY, lw=1.1, ls="--")
a.text(ns[0], hold_line - 0.5, "holding still, on the unseen ones", color=GRAY, fontsize=8, ha="left", va="top")
a.plot(ns, [u[0] for u in unseen_n], "-o", color=RED, lw=2, ms=7, mec="white", mew=1.4, label="demonstrations it never saw")
a.plot(ns, [s_[0] for s_ in seen_n], "-s", color=BLUE, lw=2, ms=6.5, mec="white", mew=1.4, label="demonstrations it trained on")
for n, u, (dx, dy, ha) in zip(ns, unseen_n, ((7, 8, "center"), (0, 8, "center"), (-7, -9, "right"), (-9, -3, "right"))):
    a.annotate(f"{u[0]:.1f}", (n, u[0]), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=8.5)
a.set_xscale("log"), a.set_xticks(ns), a.set_xticklabels([str(n) for n in ns]), a.minorticks_off()
a.set_xlabel("demonstrations trained on"), a.set_ylabel("error against the demonstrated actions")
a.set_title("The same task, more demonstrations", fontsize=10)
a.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.26))
a = axes[1]
a.axhline(hold_line, color=GRAY, lw=1.1, ls="--")
for n, u, w, c in zip(ns, unseen_n, white_n, RAMP):
    a.plot(w, u, "-o", color=c, lw=2, ms=5.5, mec="white", mew=1.2, label=str(n))
    a.annotate(str(n), (w[-1], u[-1]), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8.5)
a.legend(title="demonstrations trained on", ncol=4, frameon=False, fontsize=8, title_fontsize=8, loc="upper left", columnspacing=1.0, handlelength=1.4)
a.set_xlabel("pixels at or near white (%)"), a.set_title("The unseen ones, brightened", fontsize=10)
a.set_ylim(0, max(u.max() for u in unseen_n) * 1.2), a.set_xlim(right=white_n[0][-1] * 1.12)
fig.subplots_adjust(wspace=0.08)
save(fig, "more-demonstrations.png")
for n, s_, u, w, h in zip(ns, seen_n, unseen_n, white_n, holds_n):
    print(f"  {n:2d} demonstrations: trained-on {s_[0]:.2f} | unseen {u[0]:.2f} | holding still on its unseen frames {h:.2f} | unseen by share of pixels near white:", ", ".join(f"{wi:.0f}%: {ui:.2f}" for wi, ui in zip(w, u)))
print(f"  holding still on the unseen demonstrations: from {min(holds_n):.2f} to {max(holds_n):.2f}, drawn at {hold_line:.2f}")
