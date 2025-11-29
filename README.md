# Glance
Glance: Accelerating Diffusion Models with 1 Sample

## 📅 Updates

**16.10.2025**
- ✅ Added FLUX.1-dev LoRA training support
- ✅ Added pre-trained FLUX LoRA model example

**02.09.2025**
- ✅ Added full training for Qwen-Image and Qwen-Image-Edit

**20.08.2025**
- ✅ Added Qwen-Image-Edit LoRA trainer support

**09.08.2025**
- ✅ Add pipeline for train for < 24GiB GPU

**08.08.2025**
- ✅ Added comprehensive dataset preparation instructions
- ✅ Added dataset validation script (`utils/validate_dataset.py`)
- ✅ Freeze model weights during training


## 📦 Installation

**Requirements:**
- Python 3.10

1. Clone the repository and navigate into it:
   ```bash
   git clone https://github.com/FlyMyAI/flymyai-lora-trainer
   cd flymyai-lora-trainer
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
   # Qwen LoRA weights
   git clone https://huggingface.co/flymy-ai/qwen-image-realism-lora
   
   # FLUX LoRA weights
   git clone https://huggingface.co/flymy-ai/flux-dev-anne-hathaway-lora
   
   # Or download specific files
   wget https://huggingface.co/flymy-ai/qwen-image-realism-lora/resolve/main/flymy_realism.safetensors
   wget https://huggingface.co/flymy-ai/flux-dev-anne-hathaway-lora/resolve/main/pytorch_lora_weights.safetensors
   ```

---

## 📁 Data Preparation

### Dataset Structure for Training

The training data should follow the same format for both Qwen and FLUX models, where each image has a corresponding text file with the same name:

```
dataset/
├── img1.png
├── img1.txt
├── img2.jpg
├── img2.txt
├── img3.png
├── img3.txt
└── ...
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

### Example Data Structure

```
my_training_data/
├── portrait_001.png
├── portrait_001.txt
├── landscape_042.jpg
├── landscape_042.txt
├── abstract_design.png
├── abstract_design.txt
└── style_reference.jpg
└── style_reference.txt
```

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
4. **Dataset Size**: For good results, use at least 10-50 image-text pairs
5. **Trigger Words**: 
   - For FLUX character training: Use "ohwx woman" or "ohwx man" as trigger words
   - For Qwen training: No specific trigger words required
6. **Auto-generate descriptions**: You can generate image descriptions automatically using [Florence-2](https://huggingface.co/spaces/gokaygokay/Florence-2)

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
