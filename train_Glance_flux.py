import argparse
import logging
import os
import shutil
import copy
import datasets
import torch

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from diffusers.utils import convert_state_dict_to_diffusers
from peft.utils import get_peft_model_state_dict
from peft import TaskType
import random

from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from optimum.quanto import quantize, qfloat8, freeze
from diffusers.utils.torch_utils import is_compiled_module

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)

from peft import LoraConfig
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from image_datasets.dataset import loader
logger = get_logger(__name__, log_level="INFO")
from diffusers import FluxPipeline

# > tools -------------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()

    return args.config

# > main --------------------------------------------------------------------------------------------

def main():
    args = OmegaConf.load(parse_args())
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # > config -------------------------------------------------------------------------------------------------
    text_encoding_pipeline = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer"
    )
    quantize(flux_transformer, weights=qfloat8)
    freeze(flux_transformer)
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank*4,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    flux_transformer.to(accelerator.device)
    flux_transformer.add_adapter(lora_config)
    quantize(text_encoding_pipeline.text_encoder, weights=qfloat8)
    freeze(text_encoding_pipeline.text_encoder)
    quantize(text_encoding_pipeline.text_encoder_2, weights=qfloat8)
    freeze(text_encoding_pipeline.text_encoder_2)
    text_encoding_pipeline.to(accelerator.device)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        # step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        step_indices = torch.argmin((schedule_timesteps.unsqueeze(0) - timesteps.unsqueeze(1)).abs(), dim=1)
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
        
    vae.requires_grad_(False)
    flux_transformer.requires_grad_(False)

    flux_transformer.train()
    optimizer_cls = torch.optim.AdamW
    for n, param in flux_transformer.named_parameters():
        if 'lora' not in n:
            param.requires_grad = False
            pass
        else:
            param.requires_grad = True
            print(n)
    print(sum([p.numel() for p in flux_transformer.parameters() if p.requires_grad]) / 1000000, 'parameters')
    lora_layers = filter(lambda p: p.requires_grad, flux_transformer.parameters())

    flux_transformer.enable_gradient_checkpointing()
    optimizer = optimizer_cls(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = loader(**args.data_config)    

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    vae.to(accelerator.device, dtype=weight_dtype)
    flux_transformer, optimizer, _, lr_scheduler = accelerator.prepare(
        flux_transformer, optimizer, deepcopy(train_dataloader), lr_scheduler
    )

    initial_global_step = 0

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    train_loss = 0.0

    # > start training -------------------------------------------------------------------------------------------

    for steps, batch in enumerate(train_dataloader):
        with accelerator.accumulate(flux_transformer):
            img, prompts = batch
            with torch.no_grad():
                pixel_values = img.to(dtype=weight_dtype).to(accelerator.device)
                pixel_latents = vae.encode(pixel_values).latent_dist.sample()

                pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
                bsz = pixel_latents.shape[0]
                noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)
                
                # Determine timesteps based on strategy
                if args.Slow_LoRA and args.Fast_LoRA:
                    raise ValueError("Slow_LoRA and Fast_LoRA cannot both be True. Please configure them exclusively in the YAML.")

                if args.Slow_LoRA:
                    custom_timesteps = torch.tensor([
                        1000.0000, 986.9786, 973.2042, 958.6094, 943.1188 # Corresponds to steps 1, 3, 5, 7, and 9 (out of 50 steps)
                    ], dtype=torch.bfloat16)

                elif args.Fast_LoRA:
                    custom_timesteps = torch.tensor([
                        926.6473, 825.7017, 677.9868, 441.1997, 60.5503   # Corresponds to steps 11, 21, 31, 41, and 50 (out of 50 steps)
                    ], dtype=torch.bfloat16)

                random_idx = random.randint(0, len(custom_timesteps) - 1)
                selected_timestep = custom_timesteps[random_idx]
                timesteps = selected_timestep.repeat(bsz).to(pixel_latents.device)

                print("Selected timestep:", selected_timestep.item())

            sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
            noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

            # pack the latents.
            packed_noisy_model_input = FluxPipeline._pack_latents(
                noisy_model_input,
                batch_size=bsz,
                num_channels_latents=noisy_model_input.shape[1],
                height=noisy_model_input.shape[2],
                width=noisy_model_input.shape[3],
            )

            # latent image ids for RoPE.
            latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                bsz,
                noisy_model_input.shape[2] // 2,
                noisy_model_input.shape[3] // 2,
                accelerator.device,
                weight_dtype,
            )
            guidance_vec = torch.full(
                (bsz,),
                1,
                device=noisy_model_input.device,
                dtype=weight_dtype,
            )
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
                    list(prompts), prompt_2=list(prompts)
                )
            model_pred = flux_transformer(
                hidden_states=packed_noisy_model_input,
                timestep=timesteps / 1000,
                guidance=guidance_vec,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            model_pred = FluxPipeline._unpack_latents(
                model_pred,
                height=noisy_model_input.shape[2] * vae_scale_factor,
                width=noisy_model_input.shape[3] * vae_scale_factor,
                vae_scale_factor=vae_scale_factor,
            )
            weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)

            # flow-matching loss
            target = noise - pixel_latents
            loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            loss = loss.mean()

            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(flux_transformer.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            accelerator.log({"train_loss": train_loss}, step=global_step)
            train_loss = 0.0

            if global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    if args.Slow_LoRA:
                        ckpt_folder_name = f"checkpoint-slow-{global_step}"
                    elif args.Fast_LoRA:
                        ckpt_folder_name = f"checkpoint-fast-{global_step}"
                   
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.output_dir, ckpt_folder_name)

                try:
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                except:
                    pass
                unwrapped_flux_transformer = unwrap_model(flux_transformer)
                flux_transformer_lora_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(unwrapped_flux_transformer)
                )

                FluxPipeline.save_lora_weights(
                    save_path,
                    flux_transformer_lora_state_dict,
                    safe_serialization=True,
                )

                logger.info(f"Saved state to {save_path}")

        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
