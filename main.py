import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader


from model import ReidentificationModel
from dataset import HAM1000Dataset
from logger import LoggerService

from datetime import datetime
from tqdm import tqdm
import time

import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Medical Image Reidentification')
    parser.add_argument('--em', type=int, default=128,
                        help='size of word embeddings')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma')
    parser.add_argument('--loss_margin', type=float, default=0.2,
                        help='Triplet loss margin')
    parser.add_argument('--step_size', type=int, default=1,
                        help='Scheduler step size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='report interval (in batches)')
    args = parser.parse_args()
    return args


def evaluate(model, dataset, criterion, device):
    model.eval()

    total_loss = 0.

    with torch.no_grad():
        for (anchor, positive, negative) in dataset:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            out_anchor, out_positive, out_negative = model(
                anchor, positive, negative)
            loss = criterion(out_anchor, out_positive, out_negative)
            total_loss += loss.item()

    return total_loss / len(dataset)


def train_step(model, dataset, criterion, optimizer, device, epoch, logger, log_interval):
    model.train()

    total_loss = 0.

    for batch, (anchor, positive, negative) in enumerate(tqdm(dataset)):
        optimizer.zero_grad()

        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        out_anchor, out_positive, out_negative = model(
            anchor, positive, negative)
        loss = criterion(out_anchor, out_positive, out_negative)
        logger.log({"Training Item Loss": loss.item()})

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            avg_loss = total_loss / log_interval

            total_batches = len(dataset)

            print(
                f'Epoch {epoch} | {batch}/{total_batches} batches | avg. loss {avg_loss:5.2f}')
            logger.log({"Training Avg Loss": avg_loss})
            total_loss = 0


def get_run_config_dict(args):
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'embedding_size': args.em,
        'learning_rate': args.lr,
        'gamma': args.gamma,
        'step_size': args.step_size
    }

    return config


def main():
    args = parse_arguments()

    SEED = args.seed
    EMBEDDING_SIZE = args.em
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size
    LOG_INTERVAL = args.log_interval
    GAMMA = args.gamma
    STEP_SIZE = args.step_size
    TRIPLET_LOSS_MARGIN = args.loss_margin

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model_config = get_run_config_dict(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO: train/val/test split
    train_dataset = HAM1000Dataset('./dataset/HAM10000')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    model = ReidentificationModel(EMBEDDING_SIZE).to(device)

    optimizer = optim.Adagrad(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    criterion = nn.TripletMarginLoss(margin=TRIPLET_LOSS_MARGIN)

    logger = LoggerService(use_wandb=False)
    logger.initialize(model, model_config, None)

    best_validation_loss = None
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_directory = os.path.join(
        'models', f'model_{timestamp}_{EMBEDDING_SIZE}em')
    os.makedirs(model_directory)

    try:
        for epoch in range(1, EPOCHS + 1):
            print(f'Training epoch {epoch} out of {EPOCHS}...')
            logger.log({"Epoch": epoch})

            epoch_start_time = time.time()

            train_step(model, train_loader, criterion,
                       optimizer, device, epoch, logger, LOG_INTERVAL)

            validation_loss = 42  # TODO

            elapsed_time = time.time() - epoch_start_time

            print('-' * 89)
            print(
                f'Finished epoch {epoch} | elapsed: {elapsed_time:5.2f}s | validation loss {validation_loss:5.2f}')
            print('-' * 89)
            logger.log({"Validation Loss": validation_loss})

            if best_validation_loss is None or validation_loss < best_validation_loss:
                model_path = os.path.join(
                    model_directory, f'model_{epoch}.pth')
                with open(model_path, 'wb') as f:
                    torch.save(model, f)
                best_validation_loss = validation_loss

            logger.log({"Epoch": epoch})
            scheduler.step()
    except KeyboardInterrupt:
        print('Stopped training...')

    print('Finished training! Evaluating model on testing dataset...')
    test_loss = 42  # evaluate(testing)

    print(
        f'Finished testing! Test loss: {test_loss:.2f}')
    logger.log({"Test Final Loss": test_loss})
    logger.finish()


if __name__ == '__main__':
    main()
