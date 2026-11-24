# Toward Observability in Physical AI

Code and results for the paper **"Toward Observability in Physical AI: Offline Diagnostics for
Imitation-Learning Policies in Robotics"** (Sameer Shah).

- **Paper (PDF + DOI):** https://doi.org/10.5281/zenodo.24496519
- **Datasets:** [`sameer-s1/ball-in-the-hole`](https://huggingface.co/datasets/sameer-s1/ball-in-the-hole),
  [`sameer-s1/bread-in-toaster-dataset-v1`](https://huggingface.co/datasets/sameer-s1/bread-in-toaster-dataset-v1) (Hugging Face)
- **The failed first-run policy (2 steps):** [`sameer-s1/act-ball-in-the-hole`](https://huggingface.co/sameer-s1/act-ball-in-the-hole)

The paper turns the terms of the Action Chunking Transformer's training objective into ten offline
metrics that diagnose an imitation-learning policy from recorded files, without a robot. Every number
and figure can be reproduced from this repository plus the public datasets above.

## What is here

| Folder | Contents |
| --- | --- |
| `code/` | The ten analysis scripts. Each file's first lines say what it computes and its arguments. |
| `results/` | The `.npz` and `.json` files every test produced — the exact results analysed in the paper. |
| `training/` | The saved training configurations (`*.train_config.json`) and the loss every 200 steps (`*.loss.csv`). |

The trained model weights are not stored here (they are large and reproducible). See
**Re-run the policies** below. The two recordings used for figures — the ball-task demonstration no
policy was trained on, and the recorded rollout — are published as datasets on Hugging Face (linked
in that section).

## Check the main finding first

The saved configuration of the first run says `"steps": 2`:
<https://huggingface.co/sameer-s1/act-ball-in-the-hole/blob/main/train_config.json>

`code/theta_movement.py` rebuilds the starting parameters from the recorded seed (1000) and compares
them with that policy: no parameter has moved by more than 0.000021.

## Reproduce the numbers and figures (no robot, no GPU, no LeRobot)

Download the datasets and run two scripts:

```bash
hf download sameer-s1/ball-in-the-hole --repo-type dataset --local-dir ball
hf download sameer-s1/bread-in-toaster-dataset-v1 --repo-type dataset --include "data/*" --local-dir toaster
hf download sameer-s1/ball-unseen-demonstration --repo-type dataset --local-dir unseen-demonstration
hf download sameer-s1/ball-rollout --repo-type dataset --local-dir rollout

mkdir figures
uv run --no-project --with matplotlib --with pandas --with pyarrow --with pillow --with numpy \
    python code/make_figures.py ball rollout results figures
uv run --no-project --with numpy --with pandas --with pyarrow --with pillow \
    python code/more_metrics.py ball unseen-demonstration rollout toaster results more_metrics.json
```

`ffmpeg` must be installed. `make_figures.py` draws every figure in the paper; `more_metrics.py`
prints the numbers of Tables VI–XI, alongside `results/theta_movement.json` for the parameter
movement.

## Re-run the policies (optional; needs LeRobot 0.4.5)

The six trained checkpoints are on the Hugging Face Hub:
<https://huggingface.co/sameer-s1/physical-ai-observability-policies> (one subfolder per policy).

The exact result files are in `results/`, so the numbers can also be checked without the weights. To
reproduce a checkpoint from scratch, re-run its configuration from `training/` with the same fork of
LeRobot (the one solo-cli installed on the day of the first run), the public datasets and seed 1000:

```bash
pip install "lerobot @ git+https://github.com/GetSoloTech/solo-bot.git@85ecd003cfd4c4bf1d5360d92c6ef0724ca1fc29"
# the 20,000-step ball policy; full settings are in training/ball.train_config.json
python -m lerobot.scripts.lerobot_train --dataset.repo_id=sameer-s1/ball-in-the-hole \
    --policy.type=act --steps=20000 --batch_size=8 --seed=1000 --output_dir=out/ball-020000
cd code
python offline_predictions.py ../out/ball-020000/checkpoints/020000/pretrained_model ../ball ../rollout policy_020000.npz
```

For the toaster runs use `--dataset.repo_id=sameer-s1/bread-in-toaster-dataset-v1` with
`--dataset.episodes=[...]`: episodes 0–1 (2 demonstrations), 0–7 (8), 0–28 (29), 0–57 (58). Training
on different hardware may not be bit-identical, which is why the exact result files are provided.

## Result file names

`policy_<steps>`, `unseen_demo_<steps>`, `lighting_<steps>`, `attribution_<steps>` are tests of the
ball-task policies; `augmented_*` is the policy trained with image augmentation. `bread_<steps>` is
the toaster policy trained on 58 demonstrations, `bread_n<N>_020000` the one trained on N, and
`bread_attribution_n<N>` its input shares. `rimg_*` is the visual importance ratio (input replaced by
zeros), `rimg_mean_*` with it replaced by its training mean. `theta_movement.json` holds the parameter
movement; `more_metrics.json` the output of `more_metrics.py`.

## How this was made

The analysis code, including the code that draws the figures, was written with an AI assistant
(Claude). Every number comes from the recorded files. The toaster-task demonstrations were recorded
by the author and two teammates.

## License

Code: MIT (see `LICENSE`). The paper is under CC BY 4.0; the datasets and the policy on the Hugging
Face Hub carry their own license (Apache 2.0).
