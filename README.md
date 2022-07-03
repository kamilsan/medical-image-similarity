# Medical image similarity

## Project description

Implementation of a simple model that finds the image of most similar skin condition to the one from query image. FaceNet-based architecture with triplet loss was trained on HAM10000 dataset to realize this task. ResNet18 was selected as a backbone model. 

This was a project for Sementic Analysis of Images (2022) course.

Utilized tools and technologies:

* Python 3.8
* PyTorch 
* Weights and Biases
* venv

Dataset was split to training, testing and validation sets in proportions 80/10/10. Trained model was used to precalculate embedding for each image from testing set. Those images with embeddings were treated as a search space. Given a query image (selected from validation set), image with closest embedding vector (L2 distance) from search space was selected. 

Triplet selection procedure was based on random choice - no triplet mining was used.

## Results

Example result (cherry-picked):

![result](https://i.imgur.com/JirA2TZ.png)

For 200 randomly selected query images 56.5% images selected as closest ones were of the same class (had the same skin condition).

Results probably could have been much better, if triplet selection procedure would select hard examples, not random ones.

## Training

Results were obtained using model trained with following parameters:

* Epochs: 12
* Optimizer: Adagrad
* Initial learning rate: 0.05
* Scheduler: StepLR (gamma: 0.7, step size: 3)
* Embedding size: 128
* Batch size: 32
* Triplet loss margin: 0.2
* Seed: 42

![traning](https://i.imgur.com/WWUJmbl.png)

## Quick start

Prepare virtual environment:

```console
$ python3 -m venv venv
$ source ./venv/bin/activate
```

Install required packages:

```console
(venv) $ pip install --upgrade pip
(venv) $ pip install -r requirements.txt
```

## Usage

```console
(venv) $ python main.py
```

```
usage: main.py [-h] [--em EM] [--lr LR] [--epochs EPOCHS] [--gamma M]
               [--loss_margin LOSS_MARGIN] [--step_size STEP_SIZE]
               [--batch_size BATCH_SIZE] [--seed SEED]
               [--log_interval LOG_INTERVAL]

Medical Image Reidentification

optional arguments:
  -h, --help            show this help message and exit
  --em EM               size of word embeddings
  --lr LR               initial learning rate
  --epochs EPOCHS       number of training epochs
  --gamma M             Learning rate step gamma
  --loss_margin LOSS_MARGIN
                        Triplet loss margin
  --step_size STEP_SIZE
                        Scheduler step size
  --batch_size BATCH_SIZE
                        batch size
  --seed SEED           random seed
  --log_interval LOG_INTERVAL
                        report interval (in batches)

```

## Inference and testing

Notebook `embed.ipynb` contains some code for querying most similar images to those in testing set. Query images are choosen from validation dataset.

In order to guarantee that dataset split is the same as during training phase, same random number generator seed should be used during embedding and querying as was used before training. 

## References

Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-
source dermatoscopic images of common pigmented skin lesions. Scientific Data, 5(1).
https://doi.org/10.1038/sdata.2018.161

Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and
clustering. 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
https://doi.org/10.1109/cvpr.2015.7298682