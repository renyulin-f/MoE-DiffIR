<p align="center">
  <img src="./Figs/MoE-DiffIR.png" alt="image" style="width:1000px;">
</p>

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2407.10833)  [![Project](https://img.shields.io/badge/Project-Page-blue.svg)](https://renyulin-f.github.io/MoE-DiffIR.github.io/)  ![visitors](https://visitor-badge.laobi.icu/badge?page_id=renyulin-f/MoE-DiffIR)

This repository is the official PyTorch implementation of MoE-DiffIR (ECCV 2024).

## :bookmark: News!!!
- [x] 2024-07-15: **Arxiv version has been released.**
- [x] 2024-09-24: **Release partial of code, training and testing will come soon.**
- [x] 2024-12-27: **Release training and test code, Instructions will come soon.**

> We present MoE-DiffIR, an innovative universal compressed image restoration (CIR) method with task-customized diffusion priors. This intends to handle two pivotal challenges in the existing CIR methods: (i) lacking adaptability and universality for different image codecs, e.g., JPEG and WebP; (ii) poor texture generation capability, particularly at low bitrates. Specifically, our MoE-DiffIR develops the powerful mixture-of-experts (MoE) prompt module, where some basic prompts cooperate to excavate the task-customized diffusion priors from Stable Diffusion (SD) for each compression task. Moreover, the degradation-aware routing mechanism is proposed to enable the flexible assignment of basic prompts. To activate and reuse the cross-modality generation prior of SD, we design the visual-to-text adapter for MoE-DiffIR, which aims to adapt the embedding of low-quality images from the visual domain to the textual domain as the textual guidance for SD, enabling more consistent and reasonable texture generation. We also construct one comprehensive benchmark dataset for universal CIR, covering 21 types of degradations from 7 popular traditional and learned codecs. Extensive experiments on universal CIR have demonstrated the excellent robustness and texture restoration capability of our proposed MoE-DiffIR.

<p align="center">
  <img width="800" src="./Figs/Overview.png">
</p>

## :sparkles: Getting Start


### Prepare Datasets:
We propose a comprehensive benchmark dataset for universal compressed image restoration (CIR), covering 21 types of degradations from 7 popular traditional and learned codecs. Traditional and learned codecs include HIFIC, $$C_{SSIM}$$, $$C_{PSNR}$$. Within each codec, we apply three levels of distortions:
- **JPEG**: QF=10,15,20
- **VVC**: QP=37,42,47
- **HEVC**: QP=37,42,47
- **WEBP**: QF=1,5,10
- **HIFIC**: Mode='low', 'med', 'high'
- $$C_{SSIM}$$: Mode=1,2,3
- $$C_{PSNR}$$: Mode=1,2,3

Our CIR dataset is based on DF2K, with the original DF2K images considered as the ground truth. All compression codecs are applied to these images. Here we release the [datalink](https://drive.google.com/drive/folders/1Kn8SjJWpHITHlg5kuL1Ur7Ml-WNJJ064) of CIR dataset including 'CIR_datasets' and 'CIR_Unseen_Tasks'. The 'CIR_Dataset' branch of this dataset is shown in the following figure. The dataset comprises training and testing sets. In the training set, "DF2K_HR" contains the ground truth (GT) images, while the other folders correspond to 21 types of low-quality (LQ) images resulting from various compression distortions. The five testing sets also include folders corresponding to the 21 types of compressed LQ images, as well as folders for the high-resolution (HR) images.

### Prepare Environment:
- Python 3.9
- PyTorch 1.12.1 + cu113
Install other requiremnets:
```
pip install -r requirements.txt
```

### Inference code:
Here we release three checkpoints: (i) Only using MoE-Prompt (ii) Use Both MoE-Prompt and V2T Adapter (iii) Use MoE-Prompt, V2T Adapter and DA-CLIP degradation Prior. 
You can find three weights in this link.
Inference code: 
```
bash test.sh
```
Here you could download the klvae_ckpt from this link.
## Results
<p align="center">
  <img width="800" src="./Figs/Visual1.png">
</p>
<p align="center">
  <img width="800" src="./Figs/Visual2.png">
</p>

## Cite US
Please cite us if this work is helpful to you.

```
@article{ren2024moe,
  title={MoE-DiffIR: Task-customized Diffusion Priors for Universal Compressed Image Restoration},
  author={Ren, Yulin and Li, Xin and Li, Bingchen and Wang, Xingrui and Guo, Mengxi and Zhao, Shijie and Zhang, Li and Chen, Zhibo},
  journal={arXiv preprint arXiv:2407.10833},
  year={2024}
}
```

## Acknowledgments
The basic code is partially from the below repos.
- [StableSR](https://github.com/IceClear/StableSR)
- [StableDiffusion](https://github.com/Stability-AI/stablediffusion)


