# Glance: One Sample Distillation Model

Official PyTorch implementation of the paper:

**Glance: Accelerating Diffusion Models with 1 Sample**
<br>
[Zhuobai Dong](https://zhuobaidong.github.io/)<sup>1</sup>, 
[Rui Zhao]()<sup>2</sup>,
[Songjie Wu]()<sup>3</sup>,
[Junchao Yi]()<sup>4</sup>,
[Linjie Li]()<sup>5</sup>, 
[Zhengyuan Yang]()<sup>5</sup>, 
[Lijuan Wang]()<sup>5</sup>, 
[Alex Jinpeng Wang]()<sup>3</sup><br>
<sup>1</sup>WuHan University, <sup>2</sup>National University of Singapore, <sup>3</sup>Central South University, <sup>4</sup>University of Electronic Science and Technology of China, <sup>5</sup>Microsoft
<br>
[ArXiv](https://arxiv.org/abs/2510.14974) | [Homepage](https://zhuobaidong.github.io/Glance/) | [Model🤗](https://huggingface.co/CSU-JPG/Glance)

<img src="assets/teaser.png" alt=""/>

## 🔥News

- [Dec 1, 2025] Glance has been officially released! You can now experiment with our 1-sample distilled model.

## 📦 Installation

1. Create conda environment
   ```bash
   conda create -n glance python=3.10 -y
   conda activate glance
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the latest `diffusers` from GitHub:
   ```bash
   pip install git+https://github.com/huggingface/diffusers
   ```

4. Download pre-trained LoRA weights (optional):
   ```bash
   # Qwen Slow-LoRA weights
   wget https://huggingface.co/CSU-JPG/Glance/blob/main/glance_qwen_slow.safetensors
   
   # Qwen Fast-LoRA weights
   wget https://huggingface.co/CSU-JPG/Glance/blob/main/glance_qwen_fast.safetensors
   ```

---

## 📁 Data Preparation

### Dataset Structure for Qwen-Image and FLUX Training

In our setting, the training data consist of a single image–text pair, which still follows the required format where the image and its text description share the same filename.

```
data/
├── img1.png
├── img1.txt
```

### Dataset Structure for Qwen-Image-Edit Training

For control-based image editing, the data should be organized with separate directories for target image/caption and control image:

```
data/
├── image/           # Target image and their caption
│   ├── image_001.jpg
│   ├── image_001.txt
└── control/          # Control image
    ├── image_001.jpg
```
### Data Format Requirements

1. **Images**: Support common formats (PNG, JPG, JPEG, WEBP)
2. **Text files**: Plain text files containing image descriptions
3. **File naming**: Each image must have a corresponding text file with the same base name

### Data Preparation Tips

1. **Image Quality**: Use high-resolution images (recommended 1024x1024 or higher)
2. **Description Quality**: Write detailed, accurate descriptions of your images
3. **Auto-generate descriptions**: You can generate image descriptions automatically using [Florence-2](https://huggingface.co/spaces/gokaygokay/Florence-2)

### Quick Data Validation

You can verify your data structure using the included validation utility:

```bash
python utils/validate_dataset.py --path path/to/your/dataset
```

---
## 🎨 Inference

We provide solid 4-GPU inference code for easy multi-card sampling. You can experience our Glance model by running:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python infer_Glance_qwen.py
```

### [Glance (Qwen-Image)]
```python
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
```
### 🖼️ Sample Output - Glance-Qwen-Image

![Sample Output](./assets/qwen.png)

## 🏁 Start Training on < 24gb vram

To begin training with your configuration file (e.g., `train_lora_4090.yaml`), run:

```bash
accelerate launch train_4090.py --config ./train_configs/train_lora_4090.yaml
```
![Sample Output](./assets/Valentin_24gb.jpg)


## 🚀 Training

### Qwen-Image LoRA Training

To begin training with your configuration file, run:

```bash
accelerate launch train.py --config ./train_configs/train_lora.yaml
```

Make sure `train_lora.yaml` is correctly set up with paths to your dataset, model, output directory, and other parameters.
