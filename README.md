# SPI Encoder
Using Variantional Auto-encoder networks to understand an ensemble of heterogeneous particles from their X-ray single particle imaging data [https://arxiv.org/abs/2109.06179].

The figure below shows the architecture of the VAE neural network, consisting of a 2D CNN pattern-encoder to encode 2D patterns along with their orientation estimates into distributions of latent parameters, and a 3D transposed convolution network as volume-decoder to generate 3D intensity volumes from latent numbers. This setting allows the neural networks to learn the 3D-heterogeneity-structure-encoded latent numbers from the diffraction patterns.

![plot](https://github.com/Yulong-Zhuang/3Dvolume_recon_for_FEL_single_particel_imaging/blob/main/appendix/VAE_structure.png)

The input is 2D diffraction patterns with estimated orientations and other parameters that could be used to indicate the 3D target shapes such as gain and CLPCA.
In the paper due to the low S/N of single diffraction frames, we used Dragonfly-averaged patterns.
 
## How to use
 * Clone the repo to local.
 * Creating config file. A typical config file is shown below:
```ini
   [reconstructor]
   output_parent = VAE_test/                       #Output folder path
   output_suffix = X                               #Name prefix of the output folder
   input_path = VAE_data/VAE_cub42_ori_syn_all/    #Path to the input 2D patterns
   overwrite_output = yes                                              
 
   z_dim = 2                                       #latent space dimension
   n_info = 5                                      #additional informations (e.g. orientation, gain or CLPCA ...)
   batch_size = 10
   n_epochs = 10
   resolution_string = MD                          # Resolution options: LD: 81^3; MD: 161^3; HD: 243^2; 
```

 * Run in python as follows
   python recon.py config.ini
 
