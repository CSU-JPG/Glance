import torch
from pipeline.qwen import GlanceQwenSlowPipeline, GlanceQwenFastPipeline
from utils.distribute_free import free_pipe

repo = "CSU-JPG/Glance"
slow_pipe = GlanceQwenSlowPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.float32)
slow_pipe.load_lora_weights(repo, weight_name="glance_qwen_slow.safetensors")
slow_pipe.to("cuda")

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
).images[0].unsqueeze(0).detach().cpu()
free_pipe(slow_pipe)

fast_pipe = GlanceQwenFastPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.float32)
fast_pipe.load_lora_weights(repo, weight_name="glance_qwen_fast.safetensors")
fast_pipe.to("cuda")

image = fast_pipe(
    prompt=prompt,
    negative_prompt=" ", 
    width=1024,
    height=1024,
    num_inference_steps=5, 
    true_cfg_scale=5,
    generator=torch.Generator(device="cuda").manual_seed(0), 
    latents=latents.to("cuda", dtype=torch.float32)
).images[0]
image.save("output.png")

