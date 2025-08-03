# Real-Time Iteration Scheme for Diffusion Policy

Official codebase for ["Real-Time Iteration Scheme for Diffusion Policy"](). This repository is based on [Diffusion Policy](https://github.com/real-stanford/diffusion_policy). 

## Installation

See the [original Diffusion Policy repo](https://github.com/real-stanford/diffusion_policy) for installation. 

## Contributions

Our contributions to the repo are:
- We provide RTI-DP policies, in `policy/diffusion_unet_lowdim_rti_policy.py` and `policy/diffusion_unet_hyrbid_image_rti_policy.py` and workspace `workspace/train_diffusion_unet_lowdim_rti_workspace.py` and `workspace/train_diffusion_unet_hyrbid_rti_workspace.py`
- We provide a script for dataset scaling in `rti/scale_robomimic_dataset.py`
- We provide the evaluation file in `eval_rti.py` and config files in `config/rti`

## Evaulation with DDPM + RTI-DP
```shell
python ../eval_rti.py --config-name=eval_diffusion_rti_lowdim_workspace.yaml 
```

## Reproducing the results
`main` is a cleaner version but if you want to reproduce the result in the paper, switch to the branch `reproduce`. All parameters are provided in this branch.

We use the same checkpoints as provided by [diffusion policy](https://diffusion-policy.cs.columbia.edu/data/) for pusht, blockpush, and robomimic for RTI-DP-clip.

For RTI-DP-scale, we provide checkpoints on [huggingface](https://huggingface.co/duandaxia/rti-dp-scale).

## Citation

If you find our work useful, please consider citing [our paper]():
```bibtex
@misc{duan2025rtidp,
    title={Real-Time Iteration Scheme for Diffusion Policy},
    author={Yufei Duan and Hang Yin and Danica Kragic},
    year={2025},
}


```

## Acknowledgements

We thank the authors of [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [Consistency Policy](https://github.com/Aaditya-Prasad/Consistency-Policy/) and [Streaming Diffusion Policy](https://github.com/Streaming-Diffusion-Policy/streaming_diffusion_policy/) for sharing their codebase.