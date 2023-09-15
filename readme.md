# Ki-67 expression from H&E stains

(To be updated)

## Overview
This repository contains the Tensorflow implementation for nucleus level Ki-67 expression prediction from the optical density of H&E stained images. The directory `unet-ki67` contains the code for end-to-end U-Net training and inference. `cyclegan-ki67` and `pix2pix` are necessary codes for reproducing the GAN-based results. The GAN-based experiments were parallelized with Horovod. 