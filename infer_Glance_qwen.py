import torch
from pipeline.qwen import GlanceQwenSlowPipeline, GlanceQwenFastPipeline
from utils.distribute_free import distribute, free_pipe

repo = "CSU-JPG/Glance"
slow_pipe = GlanceQwenSlowPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.float32)
slow_pipe.load_lora_weights(repo, weight_name="glance_qwen_slow.safetensors")
distribute(slow_pipe)

prompt = "Please create a photograph capturing a young woman showcasing a dynamic presence as she bicycles alongside a river during a hot summer day. Her long hair streams behind her as she pedals, dressed in snug tights and a vibrant yellow tank top, complemented by New Balance running shoes that highlight her lean, athletic build. She sports a small backpack and sunglasses resting confidently atop her head."
latents = slow_pipe(
    prompt=prompt,
    negative_prompt=" ",
    width=1024,
    height=1024,
    num_inference_steps=5,
    true_cfg_scale=5,
    generator=torch.Generator(device="cuda").manual_seed(0), 
    output_type="latent"
).images[0]
cached_latents = latents.unsqueeze(0).detach().cpu()
free_pipe(slow_pipe)

fast_pipe = GlanceQwenFastPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.float32)
fast_pipe.load_lora_weights(repo, weight_name="glance_qwen_fast.safetensors")
distribute(fast_pipe)

loaded_latents = cached_latents.to("cuda:0", dtype=fast_pipe.transformer.dtype)
image = fast_pipe(
    prompt=prompt,
    negative_prompt=" ", 
    width=1024,
    height=1024,
    num_inference_steps=5, 
    true_cfg_scale=5,
    generator=torch.Generator(device="cuda").manual_seed(0), 
    latents=loaded_latents 
).images[0]
image.save("output.png")
