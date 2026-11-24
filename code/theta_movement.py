"""How far have the policy's parameters moved from where they started?

    <solo-cli>/.venv/bin/python theta_movement.py <out.json> <label=policy_dir> [<label=policy_dir> ...]
    e.g.   theta_movement.py results/theta_movement.json 2=<hackathon policy> 300=../outputs/checkpoints/000300 ...

The starting parameters are rebuilt by seeding the random generators as the training script does
and constructing the policy. The seed is read from the first policy folder that holds a
train_config.json (1000 here; the hackathon policy's copy of that file is on the Hub and says the
same). Each saved policy is then compared with that start, value by value: a policy trained from
another start would differ by about the size of the values themselves, not by 1e-5.

With AdamW and a learning rate of 1e-5, an early step moves a value by about 1e-5 at most, so the
hackathon policy, after 2 steps, should sit within about 2e-5 of the start.
"""
import json
import os
import sys

import numpy as np
from safetensors.numpy import load_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.random_utils import set_seed

out_path = sys.argv[1]
runs = [arg.split("=", 1) for arg in sys.argv[2:]]


def seed_of(policy_dir):
    path = policy_dir + "/train_config.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return int(json.load(f)["seed"])


seed = next(s for s in (seed_of(d) for _, d in runs) if s is not None)
cfg = PreTrainedConfig.from_pretrained(runs[0][1])
cfg.device = "cpu"
set_seed(seed)
policy = ACTPolicy(cfg)
start = {k: v.detach().numpy().astype(np.float64) for k, v in policy.state_dict().items()}
trainable = [n for n, p in policy.named_parameters() if p.requires_grad]
n_values = sum(start[k].size for k in start)
n_trainable = sum(start[k].size for k in trainable)
typical = float(sum(np.abs(start[k]).sum() for k in trainable) / n_trainable)
print(f"seed {seed} | {len(start)} tensors, {n_values:,} values, {n_trainable:,} of them trainable | mean size of a trainable value {typical:.4f}")

# For scale: how far apart are two networks that start from different seeds? Layers that load
# pretrained weights (the vision backbone) are the same whatever the seed.
set_seed(seed + 1)
other = {k: v.detach().numpy().astype(np.float64) for k, v in ACTPolicy(cfg).state_dict().items()}
apart = np.concatenate([np.abs(other[k] - start[k]).ravel() for k in trainable])
another_seed = {"seed": seed + 1, "mean_difference": float(apart.mean()), "share_identical": float((apart == 0).mean())}
print(f"a network started from seed {seed + 1} differs by {another_seed['mean_difference']:.4f} per trainable value on average; {100 * another_seed['share_identical']:.1f}% of the values are identical")

rows = []
for label, policy_dir in runs:
    assert seed_of(policy_dir) in (seed, None), f"{policy_dir} was trained with another seed"
    saved = load_file(policy_dir + "/model.safetensors")
    assert set(saved) == set(start), f"{policy_dir} has other tensors than the rebuilt policy"
    change = np.concatenate([np.abs(saved[k].astype(np.float64) - start[k]).ravel() for k in trainable])
    frozen = max(float(np.abs(saved[k].astype(np.float64) - start[k]).max()) for k in start if k not in trainable)
    row = {
        "steps": int(label),
        "mean_change": float(change.mean()),
        "median_change": float(np.median(change)),
        "largest_change": float(change.max()),
        "share_changed": float((change > 0).mean()),
        "largest_change_in_frozen_values": frozen,
    }
    rows.append(row)
    print(f"  {row['steps']:6d} steps: mean change {row['mean_change']:.3e} | median {row['median_change']:.3e} | largest {row['largest_change']:.3e}"
          f" | {row['steps']} x 1e-5 = {row['steps'] * 1e-5:.0e} | changed {100 * row['share_changed']:.1f}% | frozen values moved by {frozen:.0e}")

with open(out_path, "w") as f:
    json.dump({"seed": seed, "values": n_values, "trainable_values": n_trainable, "mean_size_of_a_value": typical, "learning_rate": float(cfg.optimizer_lr), "another_seed": another_seed, "checkpoints": rows}, f, indent=1)
print("wrote", out_path)
