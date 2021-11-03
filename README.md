
## CelebA dataset:
A wide variety of datasets suitable for use is the CelebA dataset. This is a large facial attribute dataset containing over 200,000 celebrity images, each covering a large number of variations with 40 attribute annotations.
The detailed description is available in this [Blog](https://iamgmujtaba.medium.com/generating-animated-images-gif-webp-with-cnn-and-keras-part-1-4193887cf6)

## Prerequisite
- Linux
- Python 3.6
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Clone this repo:
```bash
git clone https://github.com/iamgmujtaba/gif-celebA
cd gif-celebA
```

- Create a virtual environment by using the following commands in Ubuntu.
```bash
virtualenv --python=/usr/bin/python3.6 venv
source ./venv/bin/activate
```
- Install [TensorFlow](https://www.tensorflow.org/) and Keras and other dependencies
  - For pip users, please type the command `pip install -r requirements.txt`.

### CelebA dataset train
- Download the CelebA dataset [here](https://www.kaggle.com/jessicali9530/celeba-dataset)
- Extract the downloaded file in the data folder and the structure should look like this:

```bash
├── data/
   ├── CelebA
├── img_align_celeba/
      ├── list_attr_celeba.csv
      ├── list_bbox_celeba.csv
      ├── list_eval_partition.csv
      ├── list_landmarks_align_celeba.csv
├── utils
   ├── lib_utils.py
   ├── visdata.py
├── output
config.py
train.py
requirements.txt
```

**Data Files**
- img_align_celeba: All the face images, cropped and aligned in this folder
- list_eval_partition.csv: Recommended partitioning of images into training, validation, testing sets. Images 1-162770 are training, 162771-182637 are validation, 182638-202599 are testing
- list_bbox_celeba.csv: Bounding box information for each image. "x_1" and "y_1" represent the upper left point coordinate of bounding box. "width" and "height" represent the width and height of bounding box
- list_landmarks_align_celeba.csv: Image landmarks and their respective coordinates. There are 5 landmarks: left eye, right eye, nose, left mouth, right mouth
- list_attr_celeba.csv: Attribute labels for each image. There are 40 attributes. "1" represents positive while "-1" represents negative

## Train dataset
- Run python train code to train the CelebA dataset
```bash
python train.py
```

