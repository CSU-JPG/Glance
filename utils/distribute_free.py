import torch
import gc

def distribute(pipe, control_gpu="cuda:0"):

    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 4, f"At least 4 GPUs are required, but only {num_gpus} detected."

    # Controller GPU (GPU 0)
    pipe.text_encoder.to(dtype=torch.bfloat16).to(control_gpu)
    pipe.vae.to(control_gpu)

    # Move non-block modules of the transformer to control GPU
    for name, module in pipe.transformer.named_children():
        if name != "transformer_blocks":
            module.to(control_gpu)

    # Worker GPUs (GPU 1/2/3)
    num_worker_gpus = num_gpus - 1
    total_blocks = len(pipe.transformer.transformer_blocks)
    blocks_per_gpu = (total_blocks + num_worker_gpus - 1) // num_worker_gpus

    # Distribute transformer blocks across GPUs 1, 2, 3
    for i, block in enumerate(pipe.transformer.transformer_blocks):
        target_gpu_idx = min(i // blocks_per_gpu + 1, num_worker_gpus)
        block.to(f"cuda:{target_gpu_idx}")

def free_pipe(pipe):

    if hasattr(pipe, "transformer") and hasattr(pipe.transformer, "transformer_blocks"):
        del pipe.transformer.transformer_blocks

    if hasattr(pipe, "transformer"):
        del pipe.transformer

    if hasattr(pipe, "text_encoder"):
        del pipe.text_encoder

    if hasattr(pipe, "vae"):
        del pipe.vae

    if hasattr(pipe, "scheduler"):
        del pipe.scheduler

    if hasattr(pipe, "tokenizer"):
        del pipe.tokenizer

    del pipe

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
