# segmentation
## Segmenting nuclei in fluorescence microscopy images with deep learning.

The goal of this project is to segment nuclei from fluorescence microscopy images.
In a 3-class formulation, we try to classify each pixel of an image into either background, cell or boundary. In a boundary formulation, we predict outlines of nuclei only. In both cases we do semantic segmentation.

This code can train a CNN on multiple data sets, evaluate the model's performance and predict segmentations for new images.

A big challenge is handling overlapping and clumped nuclei. Dead cells or cells which are in the process of dividing are challeging, too.

This is my Master's Thesis which I will upload here, too.
