# Self-Supervised-Approach-for-facial-movement-based-optical-flow
This repository will contain source code and models used in our paper titled "Self-Supervised Approach for Facial Movement Based Optical Flow", available at https://arxiv.org/abs/2105.01256.

In our work, we generate an optical flow dataset specialized for faces using BP4D-Spontaneous and use it to train a CNN for learning. The pretrained models will be available after August 15th, 2022.

# Data generation
The source code and functions used to generate the data based on the paper will be found in "dataset_gen.py". This will allow the user to 

# Models and evaluation
We will add settings and instructions here to either train or evaluate optical flow. The models and code for evaluation will be available in the repository as of August 15th, 2022.

The code and models will allow the user two capabilities:
1. Train a network from scratch (using either BP4D-Spontaneous or OF-labeled dataset of your own)
2. Use the pretrained models for inference or evaluation
