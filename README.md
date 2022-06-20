# Medical image reidentification

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

Notebook `embed.ipynb` contains some code for querying most similar images to those in testing set. 
