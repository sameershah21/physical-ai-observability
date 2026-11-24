"""The visual importance ratio R_img of Zhang et al. (arXiv 2607.25516), computed for ACT policies.

    <solo-cli>/.venv/bin/python visual_importance_ratio.py ball <policy_dir> <train_dataset_dir> <out.json> [mean]
    <solo-cli>/.venv/bin/python visual_importance_ratio.py toaster <policy_dir> <dataset_repo_id> <episodes e.g. 58-63> <out.json> [mean]

Their definition: the policy is asked for its action with the unmodified inputs, with the visual
input replaced by zeros, and with the joint readings replaced by zeros. E_img and E_prop are the L2
norms of the change in the action in the two cases, and R_img = E_img / (E_img + E_prop). A value
near 100% means the action is decided by what the cameras see; near 0%, by the joint readings.
Here the action is the whole predicted chunk, and the effects are averaged over frames taken
every STRIDE-th step.

With "mean" as a last argument an input is replaced by its mean over the training data instead of
by zeros. The difference matters when zero lies far outside the range of an input: a joint that
barely moves in the demonstrations can sit hundreds of standard deviations away from zero, and a
reading of zero then changes the answer for that reason alone.
"""
import glob
import json
import sys
import tempfile

import numpy as np
import torch

STRIDE = 30
mode = sys.argv[1]
USE_MEAN = sys.argv[-1] == "mean"
e_img, e_prop = [], []

if mode == "ball":
    from act_offline import OfflinePolicy, extract_frames, load_image, table

    policy_dir, train_dir, out_path = sys.argv[2:5]
    policy = OfflinePolicy(policy_dir)
    state, _, _ = table(train_dir)
    with tempfile.TemporaryDirectory() as tmp:
        for k, path in enumerate(extract_frames(train_dir, STRIDE, tmp, "train")):
            n = k * STRIDE
            if n >= len(state):
                break
            image = load_image(path)
            base = policy.predict(image, state[n])
            no_image = policy.i_mean.expand_as(image).clone() if USE_MEAN else torch.zeros_like(image)
            no_joints = policy.s_mean.numpy().copy() if USE_MEAN else np.zeros_like(state[n])
            e_img.append(float(np.linalg.norm(base - policy.predict(no_image, state[n]))))
            e_prop.append(float(np.linalg.norm(base - policy.predict(image, no_joints))))
else:
    from safetensors.torch import load_file

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy_dir, repo_id, spec, out_path = sys.argv[2:6]
    lo, _, hi = spec.partition("-")
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

    def predict(images, joints):
        batch = {"observation.state": ((joints - s_mean) / (s_std + 1e-8)).unsqueeze(0)}
        for k in image_keys:
            m, sd = i_stats[k]
            batch[k] = ((images[k] - m) / (sd + 1e-8)).unsqueeze(0)
        with torch.no_grad():
            chunk = policy.predict_action_chunk(batch)[0]
        return (chunk * a_std + a_mean).numpy()[:, :6]  # the arm that moves in this task

    ds = LeRobotDataset(repo_id, episodes=list(range(int(lo), int(hi) + 1)), video_backend="pyav")
    for n in range(0, len(ds), STRIDE):
        item = ds[n]
        images = {k: item[k].float() for k in image_keys}
        joints = item["observation.state"].float()
        base = predict(images, joints)
        no_images = {k: (i_stats[k][0].expand_as(v).clone() if USE_MEAN else torch.zeros_like(v)) for k, v in images.items()}
        no_joints = s_mean.clone() if USE_MEAN else torch.zeros_like(joints)
        e_img.append(float(np.linalg.norm(base - predict(no_images, joints))))
        e_prop.append(float(np.linalg.norm(base - predict(images, no_joints))))

E_img, E_prop = float(np.mean(e_img)), float(np.mean(e_prop))
result = {"policy": policy_dir, "replaced_by": "training mean" if USE_MEAN else "zeros", "frames": len(e_img), "E_img": E_img, "E_prop": E_prop, "R_img_percent": 100 * E_img / (E_img + E_prop + 1e-9)}
with open(out_path, "w") as f:
    json.dump(result, f, indent=1)
print(f"{policy_dir} ({result['replaced_by']}): {len(e_img)} frames | E_img {E_img:.2f} | E_prop {E_prop:.3f} | R_img {result['R_img_percent']:.1f}%", flush=True)
