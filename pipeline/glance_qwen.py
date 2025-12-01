import torch
import numpy as np
from diffusers import DiffusionPipeline
from diffusers.pipelines.qwenimage import QwenImagePipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.utils import is_torch_xla_available
import os
from tqdm import tqdm

# --- Helper Functions (No changes) ---
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

# --- Main Pipeline Class with Fixes ---
class GlanceQwenSlowPipeline(QwenImagePipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        # 1. Input validation and setup (No changes)
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self.check_inputs(
            prompt, height, width, negative_prompt=negative_prompt, prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds, prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        # 2. Encode prompt (No changes)
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt, prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_embeds_mask, device=device,
            num_images_per_prompt=num_images_per_prompt, max_sequence_length=max_sequence_length,
        )
        
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt, prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask, device=device,
                num_images_per_prompt=num_images_per_prompt, max_sequence_length=max_sequence_length,
            )

        # 3. Prepare timesteps and latents (No changes)
        num_channels_latents = self.transformer.config.in_channels // 4
        prepared_latents = self.prepare_latents(
            batch_size * num_images_per_prompt, num_channels_latents, height, width, prompt_embeds.dtype,
            device, generator, latents,
        )
        print("Latents prepared with shape:", prepared_latents.shape)
        # Latents prepared with shape: torch.Size([1, 6032, 64])
        latents = prepared_latents.to(dtype=self.transformer.dtype)
        img_shapes = [(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)] * batch_size
        num_inference_steps = 50
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len, self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
        )
        timesteps = torch.tensor([
           1000.0000, 979.1915, 957.5157, 934.9171, 911.3354
           ], dtype=torch.bfloat16)
        timesteps = timesteps.to(device)

        num_inference_steps = 5
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
        if self.attention_kwargs is None:
            self._attention_kwargs = {}
        
        # 4. Pipeline setup
        num_worker_gpus = 3
        total_blocks = len(self.transformer.transformer_blocks)
        blocks_per_gpu = (total_blocks + num_worker_gpus - 1) // num_worker_gpus

        # 5. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # --- Positive Prompt Pass ---
                with self.transformer.cache_context("cond"):
                    current_gpu_idx = 0
                    hidden_states = self.transformer.img_in(latents).to(f"cuda:{current_gpu_idx}")
                    encoder_hidden_states = self.transformer.txt_norm(prompt_embeds).to(f"cuda:{current_gpu_idx}")
                    encoder_hidden_states = self.transformer.txt_in(encoder_hidden_states)
                    encoder_hidden_states_mask = prompt_embeds_mask.to(f"cuda:{current_gpu_idx}")
                    
                    timestep_float = timestep.to(f"cuda:{current_gpu_idx}") / 1000
                    temb = (
                        self.transformer.time_text_embed(timestep_float, hidden_states)
                        if guidance is None
                        else self.transformer.time_text_embed(timestep_float, guidance.to(f"cuda:{current_gpu_idx}"), hidden_states)
                    )
                    image_rotary_emb = self.transformer.pos_embed(
                        img_shapes, prompt_embeds_mask.sum(dim=1).tolist(), device=f"cuda:{current_gpu_idx}"
                    )

                    for block_idx, block in enumerate(self.transformer.transformer_blocks):
                        # Use robust calculation for target GPU
                        target_gpu_idx = min(block_idx // blocks_per_gpu + 1, num_worker_gpus)
                        
                        if target_gpu_idx != current_gpu_idx:
                            # [KEY FIX] Use synchronous transfer (default non_blocking=False)
                            hidden_states = hidden_states.to(f"cuda:{target_gpu_idx}")
                            encoder_hidden_states = encoder_hidden_states.to(f"cuda:{target_gpu_idx}")
                            encoder_hidden_states_mask = encoder_hidden_states_mask.to(f"cuda:{target_gpu_idx}")
                            temb = temb.to(f"cuda:{target_gpu_idx}")
                            image_rotary_emb = tuple(item.to(f"cuda:{target_gpu_idx}") for item in image_rotary_emb)
                            current_gpu_idx = target_gpu_idx

                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                            encoder_hidden_states_mask=encoder_hidden_states_mask, temb=temb,
                            image_rotary_emb=image_rotary_emb, joint_attention_kwargs=self.attention_kwargs,
                        )
                    
                    hidden_states = hidden_states.to("cuda:0")
                    temb = temb.to("cuda:0")
                    
                    hidden_states = self.transformer.norm_out(hidden_states, temb)
                    noise_pred = self.transformer.proj_out(hidden_states)

                # --- Negative Prompt Pass ---
                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        current_gpu_idx = 0
                        hidden_states = self.transformer.img_in(latents).to(f"cuda:{current_gpu_idx}")
                        encoder_hidden_states = self.transformer.txt_norm(negative_prompt_embeds).to(f"cuda:{current_gpu_idx}")
                        encoder_hidden_states = self.transformer.txt_in(encoder_hidden_states)
                        encoder_hidden_states_mask = negative_prompt_embeds_mask.to(f"cuda:{current_gpu_idx}")

                        timestep_float = timestep.to(f"cuda:{current_gpu_idx}") / 1000
                        temb = (
                            self.transformer.time_text_embed(timestep_float, hidden_states)
                            if guidance is None
                            else self.transformer.time_text_embed(timestep_float, guidance.to(f"cuda:{current_gpu_idx}"), hidden_states)
                        )
                        image_rotary_emb = self.transformer.pos_embed(
                            img_shapes, negative_prompt_embeds_mask.sum(dim=1).tolist(), device=f"cuda:{current_gpu_idx}"
                        )

                        for block_idx, block in enumerate(self.transformer.transformer_blocks):
                            target_gpu_idx = min(block_idx // blocks_per_gpu + 1, num_worker_gpus)
                            if target_gpu_idx != current_gpu_idx:
                                # [KEY FIX] Use synchronous transfer
                                hidden_states = hidden_states.to(f"cuda:{target_gpu_idx}")
                                encoder_hidden_states = encoder_hidden_states.to(f"cuda:{target_gpu_idx}")
                                encoder_hidden_states_mask = encoder_hidden_states_mask.to(f"cuda:{target_gpu_idx}")
                                temb = temb.to(f"cuda:{target_gpu_idx}")
                                image_rotary_emb = tuple(item.to(f"cuda:{target_gpu_idx}") for item in image_rotary_emb)
                                current_gpu_idx = target_gpu_idx

                            encoder_hidden_states, hidden_states = block(
                                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                                encoder_hidden_states_mask=encoder_hidden_states_mask, temb=temb,
                                image_rotary_emb=image_rotary_emb, joint_attention_kwargs=self.attention_kwargs,
                            )
                        
                        hidden_states = hidden_states.to("cuda:0")
                        temb = temb.to("cuda:0")
                        
                        hidden_states = self.transformer.norm_out(hidden_states, temb)
                        neg_noise_pred = self.transformer.proj_out(hidden_states)
                    
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # 6. Scheduler step (No changes)
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()
        
        # 7. Post-processing and VAE decoding (No changes)
        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean

            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return QwenImagePipelineOutput(images=image)

class GlanceQwenFastPipeline(QwenImagePipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        # 1. Input validation and setup (No changes)
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self.check_inputs(
            prompt, height, width, negative_prompt=negative_prompt, prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds, prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        # 2. Encode prompt (No changes)
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt, prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_embeds_mask, device=device,
            num_images_per_prompt=num_images_per_prompt, max_sequence_length=max_sequence_length,
        )
        
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt, prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask, device=device,
                num_images_per_prompt=num_images_per_prompt, max_sequence_length=max_sequence_length,
            )

        latents = latents.to(device, dtype=self.transformer.dtype)
        print("Latents prepared with shape:", latents.shape)
        img_shapes = [(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)] * batch_size
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len, self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
        )

        timesteps = torch.tensor([
             886.7053, 745.0728, 562.9505, 320.0802, 20.0000], dtype=torch.bfloat16) 
        timesteps = timesteps.to(device)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
        if self.attention_kwargs is None:
            self._attention_kwargs = {}
        
        # 4. Pipeline setup
        num_worker_gpus = 3
        total_blocks = len(self.transformer.transformer_blocks)
        blocks_per_gpu = (total_blocks + num_worker_gpus - 1) // num_worker_gpus
        # 5. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # --- Positive Prompt Pass ---
                with self.transformer.cache_context("cond"):
                    current_gpu_idx = 0
                    hidden_states = self.transformer.img_in(latents).to(f"cuda:{current_gpu_idx}")
                    encoder_hidden_states = self.transformer.txt_norm(prompt_embeds).to(f"cuda:{current_gpu_idx}")
                    encoder_hidden_states = self.transformer.txt_in(encoder_hidden_states)
                    encoder_hidden_states_mask = prompt_embeds_mask.to(f"cuda:{current_gpu_idx}")
                    
                    timestep_float = timestep.to(f"cuda:{current_gpu_idx}") / 1000
                    temb = (
                        self.transformer.time_text_embed(timestep_float, hidden_states)
                        if guidance is None
                        else self.transformer.time_text_embed(timestep_float, guidance.to(f"cuda:{current_gpu_idx}"), hidden_states)
                    )
                    image_rotary_emb = self.transformer.pos_embed(
                        img_shapes, prompt_embeds_mask.sum(dim=1).tolist(), device=f"cuda:{current_gpu_idx}"
                    )

                    for block_idx, block in enumerate(self.transformer.transformer_blocks):
                        # Use robust calculation for target GPU
                        target_gpu_idx = min(block_idx // blocks_per_gpu + 1, num_worker_gpus)
                        
                        if target_gpu_idx != current_gpu_idx:
                            # [KEY FIX] Use synchronous transfer (default non_blocking=False)
                            hidden_states = hidden_states.to(f"cuda:{target_gpu_idx}")
                            encoder_hidden_states = encoder_hidden_states.to(f"cuda:{target_gpu_idx}")
                            encoder_hidden_states_mask = encoder_hidden_states_mask.to(f"cuda:{target_gpu_idx}")
                            temb = temb.to(f"cuda:{target_gpu_idx}")
                            image_rotary_emb = tuple(item.to(f"cuda:{target_gpu_idx}") for item in image_rotary_emb)
                            current_gpu_idx = target_gpu_idx

                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                            encoder_hidden_states_mask=encoder_hidden_states_mask, temb=temb,
                            image_rotary_emb=image_rotary_emb, joint_attention_kwargs=self.attention_kwargs,
                        )
                    
                    hidden_states = hidden_states.to("cuda:0")
                    temb = temb.to("cuda:0")
                    
                    hidden_states = self.transformer.norm_out(hidden_states, temb)
                    noise_pred = self.transformer.proj_out(hidden_states)

                # --- Negative Prompt Pass ---
                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        current_gpu_idx = 0
                        hidden_states = self.transformer.img_in(latents).to(f"cuda:{current_gpu_idx}")
                        encoder_hidden_states = self.transformer.txt_norm(negative_prompt_embeds).to(f"cuda:{current_gpu_idx}")
                        encoder_hidden_states = self.transformer.txt_in(encoder_hidden_states)
                        encoder_hidden_states_mask = negative_prompt_embeds_mask.to(f"cuda:{current_gpu_idx}")

                        timestep_float = timestep.to(f"cuda:{current_gpu_idx}") / 1000
                        temb = (
                            self.transformer.time_text_embed(timestep_float, hidden_states)
                            if guidance is None
                            else self.transformer.time_text_embed(timestep_float, guidance.to(f"cuda:{current_gpu_idx}"), hidden_states)
                        )
                        image_rotary_emb = self.transformer.pos_embed(
                            img_shapes, negative_prompt_embeds_mask.sum(dim=1).tolist(), device=f"cuda:{current_gpu_idx}"
                        )

                        for block_idx, block in enumerate(self.transformer.transformer_blocks):
                            target_gpu_idx = min(block_idx // blocks_per_gpu + 1, num_worker_gpus)
                            if target_gpu_idx != current_gpu_idx:
                                # [KEY FIX] Use synchronous transfer
                                hidden_states = hidden_states.to(f"cuda:{target_gpu_idx}")
                                encoder_hidden_states = encoder_hidden_states.to(f"cuda:{target_gpu_idx}")
                                encoder_hidden_states_mask = encoder_hidden_states_mask.to(f"cuda:{target_gpu_idx}")
                                temb = temb.to(f"cuda:{target_gpu_idx}")
                                image_rotary_emb = tuple(item.to(f"cuda:{target_gpu_idx}") for item in image_rotary_emb)
                                current_gpu_idx = target_gpu_idx

                            encoder_hidden_states, hidden_states = block(
                                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                                encoder_hidden_states_mask=encoder_hidden_states_mask, temb=temb,
                                image_rotary_emb=image_rotary_emb, joint_attention_kwargs=self.attention_kwargs,
                            )
                        
                        hidden_states = hidden_states.to("cuda:0")
                        temb = temb.to("cuda:0")
                        
                        hidden_states = self.transformer.norm_out(hidden_states, temb)
                        neg_noise_pred = self.transformer.proj_out(hidden_states)
                    
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # 6. Scheduler step (No changes)
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()
        
        # 7. Post-processing and VAE decoding (No changes)
        # torch.cuda.empty_cache()
        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return QwenImagePipelineOutput(images=image)






