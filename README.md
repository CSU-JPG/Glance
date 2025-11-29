# Glance

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
[arXiv](https://arxiv.org/abs/2510.14974) | [homepage](https://zhuobaidong.github.io/Glance/) | [Model🤗](https://huggingface.co/spaces/Lakonik/pi-Qwen)

<img src="assets/teaser.jpg" alt=""/>

## 🔥News

- [Nov 7, 2025] [ComfyUI-piFlow](https://github.com/Lakonik/ComfyUI-piFlow) is now available! Supports 4-step sampling of Qwen-Image and Flux.1 dev using 8-bit models on a single consumer-grade GPU, powered by [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

**02.09.2025**
- ✅ Added full training for Qwen-Image and Qwen-Image-Edit

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
   git clone https://huggingface.co/flymy-ai/qwen-image-realism-lora
   
   # Qwen Fast-LoRA weights
   git clone https://huggingface.co/flymy-ai/flux-dev-anne-hathaway-lora
   
   # Or download specific files
   wget https://huggingface.co/flymy-ai/qwen-image-realism-lora/resolve/main/flymy_realism.safetensors
   wget https://huggingface.co/flymy-ai/flux-dev-anne-hathaway-lora/resolve/main/pytorch_lora_weights.safetensors
   ```

---

## 📁 Data Preparation

### Dataset Structure for Qwen-Image and FLUX Training

The training data should follow the same format for both Qwen and FLUX models, where each image has a corresponding text file with the same name:

```
dataset/
├── img1.png
├── img1.txt
```

### Dataset Structure for Qwen-Image-Edit Training

For control-based image editing, the dataset should be organized with separate directories for target images/captions and control images:

```
dataset/
├── images/           # Target images and their captions
│   ├── image_001.jpg
│   ├── image_001.txt
│   ├── image_002.jpg
│   ├── image_002.txt
│   └── ...
└── control/          # Control images
    ├── image_001.jpg
    ├── image_002.jpg
    └── ...
```
### Data Format Requirements

1. **Images**: Support common formats (PNG, JPG, JPEG, WEBP)
2. **Text files**: Plain text files containing image descriptions
3. **File naming**: Each image must have a corresponding text file with the same base name

### Text File Content Examples

**For FLUX character training (portrait_001.txt):**
```
ohwx woman, professional headshot, studio lighting, elegant pose, looking at camera
```

**For Qwen landscape training (landscape_042.txt):**
```
Mountain landscape at sunset, dramatic clouds, golden hour lighting, wide angle view
```

**For FLUX portrait training (abstract_design.txt):**
```
ohwx woman, modern portrait style, soft lighting, artistic composition
```

### Data Preparation Tips

1. **Image Quality**: Use high-resolution images (recommended 1024x1024 or higher)
2. **Description Quality**: Write detailed, accurate descriptions of your images
3. **Consistency**: Maintain consistent style and quality across your dataset
4. **Auto-generate descriptions**: You can generate image descriptions automatically using [Florence-2](https://huggingface.co/spaces/gokaygokay/Florence-2)

### Quick Data Validation

You can verify your data structure using the included validation utility:

```bash
python utils/validate_dataset.py --path path/to/your/dataset
```

This will check that:
- Each image has a corresponding text file
- All files follow the correct naming convention
- Report any missing files or inconsistencies

---

## 🏁 Start Training on < 24gb vram

To begin training with your configuration file (e.g., `train_lora_4090.yaml`), run:

```bash
accelerate launch train_4090.py --config ./train_configs/train_lora_4090.yaml
```
![Sample Output](./assets/Valentin_24gb.jpg)


## 🏁 Training

# Qwen Models Training

## Qwen-Image LoRA Training

To begin training with your configuration file (e.g., `train_lora.yaml`), run:

```bash
accelerate launch train.py --config ./train_configs/train_lora.yaml
```

Make sure `train_lora.yaml` is correctly set up with paths to your dataset, model, output directory, and other parameters.
