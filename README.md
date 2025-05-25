<div align="center">

# Your Diffusion Model is an Implicit Synthetic Image Detector

This repository contains the official implementation of our paper:  
**"Your Diffusion Model is an Implicit Synthetic Image Detector"**

</div>

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🚀 Getting Started

To evaluate the method, run the following script:

bash evaluate.sh

---

## 📦 Pretrained Checkpoint

Download the pretrained model checkpoint here:  
https://drive.google.com/file/d/1Cjfrdbb6GBrf7omPdE_3c8IYVSPK977A/view?usp=sharing

---

## 📁 Evaluation Data

Sample evaluation data is available here:  
https://drive.google.com/file/d/1gv3hEL4Rup7GUxCQ43-YKRMhsEd_F92f/view?usp=sharing

---

## 🔍 Using Your Own Data

To evaluate on your own data:

1. **Generate Latents**  
   Run the script below to compute latents (you may want to modify it accordingly):

   bash data_preparation/compute_latent.sh 

2. **Run Inference**  
   Use the script below to make predictions:

   bash inference_custom_dataset.sh

---
