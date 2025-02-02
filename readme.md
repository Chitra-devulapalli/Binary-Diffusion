# Overview
This repository contains my implementation of denoising diffusion probabilistic models (DDPM) applied to a simplified two-pixel image space. The project explores the fundamentals of diffusion models, training and sampling methodologies, and the visualization of diffusion dynamics in low-dimensional image distributions.

By focusing on a two-pixel world, we can fully visualize the probability density function (PDF) of the learned image distributions, allowing a deeper understanding of how generative models learn structure in data. The implementation includes training a conditional UNet, optimizing diffusion processes, and visualizing diffusion trajectories with classifier-free guidance.

## Implemented Features
1. Denoising Diffusion Probabilistic Models (DDPM)
* Implemented the forward diffusion process by gradually adding Gaussian noise to two-pixel images.
* Modeled the reverse process using a learned neural network to denoise and reconstruct images.
  
2. Conditional UNet for Noise Prediction
* Built a lightweight UNet architecture to predict noise at each diffusion step.
* Incorporated sinusoidal Fourier embeddings for encoding diffusion time-steps.
* Added class conditioning to guide denoising toward different categories.
  
3. Beta Scheduling and Variance Estimation
* Implemented sinusoidal beta scheduling for improved stability during training.
* Computed cumulative alpha values and posterior variance for diffusion modeling.

4. Training and Sampling
* Trained the model using an L2 loss between predicted and actual noise.
* Implemented both classifier guidance and classifier-free guidance for controlled image generation.
* Sampled denoised images using reverse diffusion with a stochastic noise term.

5. Visualizing Diffusion Dynamics
* Generated diffusion trajectories overlaid on the estimated probability density function (PDF).
  ![Screenshot from 2025-02-01 19-23-00](https://github.com/user-attachments/assets/3858c5ca-1cc7-4149-a272-a421778a18b5)

* Experimented with different diffusion timesteps (T=5,10,25,50) to analyze model convergence.
  ![Screenshot from 2025-02-01 19-23-44](https://github.com/user-attachments/assets/5f6a2561-9f0a-424f-813d-eb52b1813f28)

* Compared guided vs. unguided denoising to observe the effect of conditioning.
  ![Screenshot from 2025-02-01 19-24-00](https://github.com/user-attachments/assets/2292d77f-8980-40f2-a180-f9e680a8a5ba)
