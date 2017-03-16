# segmentation
## We work on this nice project.

The goal of this project is to segment nuclei from fluorescence microscopy images.
We try to classify each pixel of an image into either background, cell or boundary. Thus, we do semantic segmentation.

This code can train a CNN on multiple data sets, evaluate the model's performance and predict segmentations for new images.

A big challenge is handling overlapping and close nuclei as well as nuclei that have a weird shape. Dead cells or cells which are in the process of dividing may have weird shapes.
