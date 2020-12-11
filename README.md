# sdss_CNN

Deep convolutional neural network for predicting a galaxy's spin based on SDSS photometry (RGB images).

This directory using kinematic information from [MaNGA](https://www.sdss.org/surveys/manga/), along with photometry from the Sloan Digital Sky Survey ([SDSS](https://classic.sdss.org/dr7/)), to train a convolutional neural network to predict a galaxy's spin. The aim of the game here, is to train a convolutional neural network to predict a galaxy's spin based on an image of the galaxy. MaNGA observations (resulting in kinematic information) are expensive and hence we only have 1000s of galaxies with this information. On the other hand, we have millions of SDSS galaxy images, so extrapolating kinematic properties would be useful.

## Data

### MaNGA 
MaNGA data is taken from the Data Analysis Pipeline ([DAP](https://www.sdss.org/dr15/manga/manga-analysis-pipeline/)) which provides stellar and (ionized) gas velocity fields. Velocity fields are found by fitting the stellar continuum, from which the spin parameter λ<sub>R</sub> is estimated. λ<sub>R</sub> [introduced here](https://arxiv.org/abs/astro-ph/0703531) is a measure of _spin_ for a galaxy, by computing the light weighted average of ordered rotation divided by dispersion (random motion). 

### SDSS photometry (i.e. RGB image)
SDSS has created the most detailed three-dimensional maps of the Universe (_so far_) with deep multi-color images of galaxies covering one third of the sky. The vast majority of MaNGA galaxies have been imaged by SDSS, which we can use as input into our convolutional neural network. The exact resolution of an image (relative to the galaxy size) is variable (since it is dependent on the distance to the galaxy). 

### Pre-processing
