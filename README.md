
# Direct Preference Optimization (DPO) for GPT-2 Fine-Tuning

## Project Overview
This project fine-tunes GPT-2 using **Direct Preference Optimization (DPO)** to align the model's responses with user preferences. The training utilizes **UltraFeedback**, a dataset containing human feedback on model-generated responses.

## Dataset
The original dataset contained only a **train split** with **61,917** samples and required preprocessing. The dataset included additional metadata fields such as **ratings and models**, which were not necessary for training. To refine it, only **three key fields** were extracted: `prompt`, `chosen`, and `rejected`. After cleaning, the dataset was split into **train (80%) and test (20%)**, resulting in **49,733** training samples and **13,184** test samples.

---

## Model Details
- **Pretrained Model**: `GPT-2`
- **Training Configuration**:
  - **Device**: CUDA (GPU)
  - **Optimizers Tested**: AdamW, RMSprop
  - **Batch Sizes**: 4, 8
  - **Learning Rates**: `5e-4`, `1e-3`, `2e-3`
  - **Beta Values**: `0.1`, `0.2`, `0.3`
  - **Training Steps**: 500
  - **Evaluation Strategy**: Every 500 steps

---

## Training & Evaluation Results
Performance results of trainings configurations are shown below.

- **Best Model Configuration**:
  - **Learning Rate**: `0.002`
  - **Batch Size**: `8`
  - **Optimizer**: AdamW
  - **Beta**: `0.3`
  - **Training Steps**: `500`

---

sorry i cant deploy in my Hugging Face sorry 
