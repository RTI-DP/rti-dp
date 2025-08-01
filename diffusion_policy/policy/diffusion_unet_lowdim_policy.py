from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import matplotlib
import numpy as np
import cProfile
import pstats

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

##python eval.py --checkpoint data/default/epoch\=0550-test_mean_score\=0.969.ckpt --output_dir data/todele 

class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDIMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs
        self.init = True

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        ##NOTE: num_inference_steps is set to 100
        print("denoising type", type(self.noise_scheduler))
        self.num_inference_steps = num_inference_steps

        print(self.num_inference_steps)
        # self.n_action_steps = 20
        # print("action steps: ", self.n_action_steps)    


    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            past_acs=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        
        profiler = cProfile.Profile()
        profiler.enable()
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # if self.init and past_acs is not None:
            # trajectory[:,:8,:2] = past_acs

        # set step values
        # scheduler.num_inference_steps=100
        scheduler.set_timesteps(self.num_inference_steps)

        prevtraj = trajectory

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
            prevtraj = trajectory

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
            # print(t)
            # print(trajectory.shape)


        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        profiler.disable()
        with open('profile.prof', 'w') as f:
            ps = pstats.Stats(profiler, stream=f)
            ps.sort_stats(pstats.SortKey.TIME)
            ps.print_stats()
        return trajectory
    

    def nr_conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            past_acs=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        
        profiler = cProfile.Profile()
        profiler.enable()
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # if self.init and past_acs is not None:
            # trajectory[:,:8,:2] = past_acs

        # set step values
        scheduler.num_inference_steps=100
        scheduler.set_timesteps(self.num_inference_steps)

        prevtraj = trajectory

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
            prevtraj = trajectory

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
            print(t)
            print(trajectory.shape)

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        profiler.disable()
        with open('profile.prof', 'w') as f:
            ps = pstats.Stats(profiler, stream=f)
            ps.sort_stats(pstats.SortKey.TIME)
            ps.print_stats()
        return trajectory
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key

        unnormalized pred version
        """

        assert 'obs' in obs_dict
        #assert 'past_action' not in obs_dict # not implemented yet

        past_acs = None
        if 'past_action' in obs_dict:
            past_acs = self.normalizer['action'].normalize(obs_dict['past_action'])
            print("past action: ", past_acs.shape)
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            past_acs=past_acs,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        a, b, c = naction_pred.shape

        print(naction_pred.shape)

        # print("predicted gripper:", naction_pred[0,:,-1])
        # if c==20:
        #     print("predicted gripper:", naction_pred[0,:,9])
        # print("no norm", naction_pred[0,0,:])

        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        # print("after norm", action_pred[0,0,:])

        # action_pred[:,:,-1] = torch.where(action_pred[:,:,-1] < 0., -1., 1.)
        # if c == 20:
        #     action_pred[:,:,9] = torch.where(action_pred[:,:,-1] < 0., -1., 1.)

        # action_pred[:,:,-1]=action_pred[:,:,-1]*10.0
        # if c == 20:
        #     action_pred[:,:,9]=action_pred[:,:,9]*10.0

        # # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            #in this loop
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            print("result shape", action_obs_pred.shape)
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
