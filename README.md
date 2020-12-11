# sdss_CNN

Deep convolutional neural network for predicting a galaxy's spin based on SDSS photometry (RGB images).

This directory using kinematic information from [MaNGA](https://www.sdss.org/surveys/manga/), along with photometry from the Sloan Digital Sky Survey ([SDSS](https://classic.sdss.org/dr7/)), to train a convolutional neural network to predict a galaxy's spin. The aim of the game here, is to train a convolutional neural network to predict a galaxy's spin based on an image of the galaxy. MaNGA observations (resulting in kinematic information) are expensive and hence we only have 1000s of galaxies with this information. On the other hand, we have millions of SDSS galaxy images, so extrapolating kinematic properties would be useful.

## Data

### MaNGA 
MaNGA data is taken from the Data Analysis Pipeline ([DAP](https://www.sdss.org/dr15/manga/manga-analysis-pipeline/)) which provides stellar and (ionized) gas velocity fields. Velocity fields are found by fitting the stellar continuum, from which the spin parameter λ<sub>R</sub> is estimated. λ<sub>R</sub> [introduced here](https://arxiv.org/abs/astro-ph/0703531) is a measure of _spin_ for a galaxy, by computing the light weighted average of ordered rotation divided by dispersion (random motion). 

### SDSS photometry (i.e. RGB image)
SDSS has created the most detailed three-dimensional maps of the Universe (_so far_) with deep multi-color images of galaxies covering one third of the sky. The vast majority of MaNGA galaxies have been imaged by SDSS, which we can use as input into our convolutional neural network. The exact resolution of an image (relative to the galaxy size) is variable (since it is dependent on the distance to the galaxy). 

Galaxy images are pulled from the SDSS database on [sciserver](https://www.sciserver.org/) see in `./data/on_sciserver/`. Each image downloaded is 424 x 424 pixels in size where each pixel is set to be 0.02R<sub>e</sub> (elliptical petrosian half-light radius) of the target galaxy (i.e. image covers 8.48R<sub>e</sub> for each galaxy). 

### Pre-processing for CNN
For input into the CNN, we downsample all galaxy images to be size (80, 80, 3) (to avoid fitting noise), and, normalised so pixel values (in each channel) range [0, 1]. We have 6437 galaxies, both with SDSS images and MaNGA kinematic information which we use to train, validate, and, test our network. We split (70, 15, 15)% so that:

| Sample  | Count |
| ------------- | ------------- |
| Training | 4505 |
| Test | 966 |
| Validation | 966 |

Galaxy images are augmented for the training sample, where during each epoch of the training images are randomly zoomed (±25%), rotated (±45 degrees), flipped (both horizontally and vertically), and, shifted both vertically and horizontally (by 5 per cent). 

## CNN

### Architecture

![diagram](./plots/cnn_schematic.png)
### 