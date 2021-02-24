# Deep-Dream-using-Tensorflow
 
Deep Dream is a computer vision program created by Google which uses convolutional neural networks to find and enhance patterns in images thus creating a dream like hallucinogenic apppearance in the deliberately over_processed images.
We are using [inception5h]((https://storage.googleapis.com//download.tensorflow.org//models//inception5h.zip)) from Tensorflow for the problem.

### Code and Resources

**Language:** Python 3.8

**Modules and Libraries:** matplotlib, numpy, tensorflow, urllib, zipfile and PIL.

**Dataset:** Google

**Keywords:** Deep learning, Computer vision, Neural Networks, Image Enhancer

**Step 1:** Import all required Libraries and Modules. Download and Extract the data as well.

**Step 2:** Download and extract the Google's Pre-trained Neural Network.

**Step 3:** Start with gray image with little noise.

**Step 4:** Create a Tensorflow Session and load the model plus define the input tensor.

**Step 5:** Define the hyperparameters of the neural network layers.

**Step 6:** Write a helper function for TF graph visualization.

**Step 7:** Pick a layer to enhance our image here we are using "mixed4d_3x3_bottleneck_pre_relu".

**Step 8:** Pick a feature channel to visualize.

**Step 9:** Open the image and apply gradient ascent to that layer.

**Step 10:** Output Deep Dream image using Matplotlib.

**Results**

![drugs]()

**References**

1. https://www.youtube.com/watch?v=MrBzgvUNr4w
2. https://github.com/llSourcell/deep_dream_challenge
