# Self-Supervised-Approach-for-facial-movement-based-optical-flow
This repository contains the source code and models used in our paper titled "Self-Supervised Approach for Facial Movement Based Optical Flow", paper available at https://ieeexplore.ieee.org/document/9854154 and preprint at https://arxiv.org/abs/2105.01256.

In our work, we generate an optical flow dataset specialized for faces using BP4D-Spontaneous and use it to train a CNN for learning. The requirements.txt file contains the dependencies required to train or evaluate our models in the same manner as the paper.

If you are only interested in data generation, you may remove all the Keras and tensorflow related dependencies in the requirements file.

# Data generation
"facialof_datasetgen.py" contains the source code needed to replicate our dataset generation process using BP4D-Spontaneous and CK+.
The "generate_new_landmarks" variable, set to True, will use OpenFace to detect facial keypoints and store the results in .csv files.
The method assumes that the file structure for your dataset is as follows, similar to how BP4D-Spontaneous and CK+ are structured:

1) One **root directory** that contains a Subject directory with all **subject directories**.
2) An individual **subject directory** contains all **sequence directories** for that particular subject.
3) An individual **sequence directory** will contain all the images of that particular sequence (other non image files are ok, the code will read only image files

If "generate_new landmarks" is set to False, the root directory should also contain a Landmarks directory with identical directories inside. The only difference is that the sequences will contain landmarks .txt files (please see 'Keypoint detection' section for more details).
## Directory structure
The user will need to manually define the paths right after the function definitions (lines 477-483). 
The directories to be defined are:
1) data_root_path: root path of your dataset
2) images_path: path to directory containing all subjects, within the data_root_path
3) save_path: path to directory in which the new optical flow data is to be saved
4) openface_dir: if "generate_new_landmarks" is set to True, this is path to your OpenFace build directory          

## Keypoint detection
If you require to generate landmarks, with "generate_new_landmarks" set to True, the code uses OpenFace 2.2.0 by Tadas Baltrusaitis, available at https://github.com/TadasBaltrusaitis/OpenFace/wiki. Complete instructions to install and setup are provided in the Github repo. You'll need to provide this path and make sure that the code can run bash operations to use OpenFace.

If you have your own keypoint data, the .txt files should be located and named in the exact same way as the image files, appended with '_landmarks.txt' (e.g. image name is '0001.png', keypoint file name should be '0001_landmarks.txt'.

The code will extract (x, y) coordinates from the two column .txt file for each image, with the first column containing x coordinates and second column containing y coordinates. 

If you have the keypoints structured in a different manner, you can adapt the small section which reads the landmarks as it suits you.

# Models and evaluation
This section is related to training and evaluating the CNNs for optical flow. The codes are "facialof_train.py" and "facialof_evaluate.py".

The code and models allows the user two capabilities:
1) Train a network from scratch (using either BP4D-Spontaneous or OF-labeled dataset of your own)
2) Use pretrained models for inference and evaluation

The pretrained weights for our experiment are 'cp.ckpt', available in each of exp1, exp2, and exp3. These folders are named identically to the naming convention found in the paper. The directories can be downloaded directory from https://drive.google.com/drive/folders/1CiLTiazhmwQAZ9soxijAvJnXJ5ucZotK?usp=sharing 
