# SPI Encoder
Using encoder-decoder networks in combination with likelihood information to understand an ensemble of heterogeneous particles from their X-ray single particle imaging data.

## Installation
Other than the standard packages, one needs `cupy` and `CuPADMAN` as prerequisites.

Create conda environment.

```sh
$ git clone git@github.com:AyyerLab/SPIEncoder.git
$ cd SPIEncoder
$ pip install -e . # Note dot at end
```

## Usage

 * Create configuration file with an `[encoder]` section
```ini
[encoder]
in_detector_file = data/det.h5
num_div = 6
loss_type = euclidean
```

 * Run in python as follows
```python
import cupy as cp
from SPIEncoder import LossCalculator

cp.cuda.Device(0).use() # Choose device number

lc = LossCalculator('config.ini') # Path to config file
lc.calc_loss(img, model)

# img.shape == (lc.det.num_pix,)
# model.shape == (lc.size, lc.size, lc.size)
```

## Notes and To-Do's
 * Currently only Euclidean loss is available. Likelihood-based loss for individual frames should be included.
 * Parsing of class averages. Sparse frames can already be parsed.
 * Creating of detector file for class averages.
 * Implementation of actual VAE network, trained with this loss function
 * Optimization of loss calculation by aggregating frames with the same/similar model

