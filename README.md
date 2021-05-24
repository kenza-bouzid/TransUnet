# TransUNet

This repo reproduces the results of  [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2102.04306.pdf) as a final project for the course Deep Learning in Data Science DD2424 @ KTH (Royal Institute of Technology)

A demo of TransUnet is available in [this colab notebook](https://colab.research.google.com/github/KenzaB27/TransUnet/blob/main/TransUNet_demo.ipynb#scrollTo=QwBIRuD4tAfc).

A pretrained model trained on synapse dataset can be downloaded via [this link](https://drive.google.com/file/d/1ugXdSGGDJaOM-rOx_boQYoO71tTSe9k6/view?usp=sharing).

Authors: *Agnieszka Miszkurka, Kenza Bouzid, Tobias Höppe*

## Environment

The project is implemented with Tensorflow 2. med-py library is used for medical image segmentation evaluation (Hausdorf Distance and Dice Score).

Prepare an virtual environment with python>=3.6, and then use the following command line for the dependencies.

```bash
pip install -r requirements.txt
```

## Data 

The  experiments were conducted on Synapse multi-organ segmentation dataset.

* Access to the synapse multi-organ dataset:

  * Sign up in the [official Synapse website](https://www.synapse.org/#!Synapse:syn3193805/wiki/) and download the dataset. Convert them to numpy format, clip the images within [-125, 275], normalize each 3D image to [0, 1], and extract 2D slices from 3D volume for training cases while keeping the 3D volume in h5 format for testing cases.
  * It is possible to request the preprocessed dataset from the  [original repo](https://github.com/Beckschen/TransUNet/edit/main/datasets/README.md) authors.
  * Set up a Google Cloud Project to store your data in a bucket.
  * Convert the data from numpy to TfRecords (Tensorflow’s binarystorage format) to speed up training and enable parallel data reading from disk. We provide a data parsing pipeline to write and read TfRecords as a TFDataset in the module ``data_processing``. A guide notebook is available under ``experiments/create_tfds_records.ipynb``.

* The directory structure of the whole project is as follows:

  ```bash
  ├───data
  │   ├───synapse-train-224
  │   		├── record_0.tfrecords
  │   		└── *.tfrecords
  |   ├───synapse-test-224
  │   		├── case0001.tfrecords
  │   		└── *.tfrecords
  │   ├───test_vol_h5
  │   		├── case0001.npy.h5
  │   		└── *.npy.h5
  │   └───train_npz
  │   		├── case0005_slice000.npz
  │   		└── *.npz
  ├── TransUNet
      ├───data_processing
      │   ├───dataset_synapse.py
      │   ├───data_parser.py
      │   └───__init__.py
      ├───experiments
      │	├───config.py
      │   ├───create_tfds_records.ipynb
      │   ├───data_exploration.ipynb
      │   └───__init__.py
      ├───models
      │   ├───decoder_layers.py
      │   ├───encoder_layers.py
      │   ├───resnet_v2.py
      │   ├───transunet.py
      │   ├───utils.py
      │   └───__init__.py
      ├───synapse_ct_scans
      │   ├───case0022.tfrecords
      │   ├───case0025.tfrecords
      │   └───case0029.tfrecords
      └───utils
          ├───evaluation.py
          ├───visualize.py
          └───__init__.py
  ```
  
  **We provide some synapse CT scans written as Tfrecords for testing.**

## Train/Test

#### Train

We provide 4 different architectures that can be selected from the config file in ``experiments`` module:

* B16-None: ``config.get_b16_none()``
* B16-CUP: ``config.get_b16_cup()``
* B16+R50-CUP: ``config.get_r50_b16()``
* Transunet`:``config.get_transunet()``

An instance of the model can be created and compiled/ trained as follows:

```python
from models.transunet import transunet
from data_processing.data_parser import DataReader
from experiments.config import get_transunet 

## Prepare data 
dr = DataReader(src_path="YOUR_SRC_PATH", height=224, width=224, depth=3)
training_dataset, validation_dataset = dr.get_dataset_training(image_size=224)
## Train Model 
config = get_transunet()
transunet = TransUnet(config)
transunet.compile()
history = transunet.train_validate(training_dataset, validation_dataset, save_path, epochs=150)
```

 We provide an example for transUNet that can generalized to the other architectures.

#### Test

Inference and Test can be performed both visually and quantitatively by computing the Dice Score of the predicted label maps.

First, write and save the volumes test data as list:

```python
from data_processing.data_parser import DataWriter
dw = DataWriter(src_path="YOUR_SRC_PATH", height=224, width=224, depth=3)
test_dataset = dw.write_test_list()
```

Then, perform Inference as follows:

```python
from utils.evaluation import inference
inference(test_dataset, model.model)
```

Visualize the segmentation masks as follows:

```python
from utils.visualize import visualize_non_empty_predictions
from data_processing.data_parser import DataReader
dr = DataReader(src_path="YOUR_SRC_PATH", height=224, width=224, depth=3)
test_dataset = dr.get_test_data()

for i, img_lab in enumerate(test_dataset.take(32)):
  img = img_lab[0]
  lab = img_lab[1]
  visualize_non_empty_predictions(img, lab, models)
```

For recall ``TransUNet_demo.ipynp`` notebook provides an end to end demo that loads a pretrained transUNet model and visualizes the predicted segmentation masks. It is also available as a  [colab notebook](https://colab.research.google.com/github/KenzaB27/TransUnet/blob/main/TransUNet_demo.ipynb#scrollTo=QwBIRuD4tAfc).

## References 

* [TransUNet](https://github.com/Beckschen/TransUNet)
* [vit-keras](https://github.com/faustomorales/vit-keras)
* [ResNetV2](https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/applications/resnet_v2.py#L28-L56)

