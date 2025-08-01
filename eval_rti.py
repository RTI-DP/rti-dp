"""
Usage:
python eval_rti.py --config-name=eval_diffusion_rti_lowdim_workspace.yaml 
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from omegaconf import OmegaConf

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config','rti'))
)

def main(target_cfg: OmegaConf):
    OmegaConf.resolve(target_cfg)

    checkpoint = target_cfg.checkpoint
    output_dir = target_cfg.output_dir

    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(target_cfg._target_)

    OmegaConf.set_struct(cfg, False)
    cfg.n_action_steps = target_cfg.n_action_steps
    cfg.policy.n_action_steps = target_cfg.n_action_steps + cfg.n_latency_steps
    cfg.task.env_runner.n_action_steps = target_cfg.n_action_steps
    cfg.policy._target_ = target_cfg.policy._target_
    cfg.policy.noise_scheduler = target_cfg.policy.noise_scheduler
    cfg.policy.steps = target_cfg.policy.steps
    print(target_cfg.policy.steps)
    cfg.policy.has_discrete_action = target_cfg.policy.discrete_action.has_discrete_action
    if cfg.policy.has_discrete_action:
        cfg.policy.scale = target_cfg.policy.discrete_action.scale
        cfg.policy.factor = target_cfg.policy.discrete_action.factor
        
    if "variance_type" in cfg.policy:
        del cfg.policy["variance_type"]

    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

    # Save steps and checkpoint to a separate param config file
    param_config = {
        "steps": OmegaConf.to_container(cfg.policy.steps, resolve=True),
        "checkpoint": checkpoint
    }
    param_config_path = os.path.join(output_dir, 'param_config.json')
    with open(param_config_path, 'w') as f:
        json.dump(param_config, f, indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
