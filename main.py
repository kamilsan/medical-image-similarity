import argparse
import torch
import torch.nn as nn
import torch.optim as optim
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
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--log_interval', type=int, default=200,
                        help='report interval (in batches)')
    args = parser.parse_args()
    return args


def evaluate(model, dataset, criterion):
    model.eval()

    total_loss = 0.

    with torch.no_grad():
        for (anchor, positive, negative) in dataset:
            out_anchor, out_positive, out_negative = model(
                anchor, positive, negative)
            loss = criterion(out_anchor, out_positive, out_negative)
            total_loss += loss.item()

    return total_loss / len(dataset)


def train_step(model, dataset, criterion, optimizer, epoch, logger, log_interval):
    model.train()

    total_loss = 0.

    for batch, (anchor, positive, negative) in enumerate(tqdm(dataset)):
        optimizer.zero_grad()

        out_anchor, out_positive, out_negative = model(
            anchor, positive, negative)
        loss = criterion(out_anchor, out_positive, out_negative)
        logger.log({"Training Item Loss": loss.item()})

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            avg_loss = total_loss / log_interval

            total_batches = len(dataset.dataset)

            print(
                f'Epoch {epoch} | {batch}/{total_batches} batches | avg. loss {avg_loss:5.2f}')
            logger.log({"Training Avg Loss": avg_loss})
            total_loss = 0


def main():
    args = parse_arguments()

    SEED = 42

    EMBEDDING_SIZE = 128
    LEARNING_RATE = 1e-4
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-5
    BATCH_SIZE = 8
    TRIPLET_LOSS_MARGIN = 0.2
    EPOCHS = 20
    LOG_INTERVAL = 200

    model_config = {}  # TODO

    torch.manual_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = HAM1000Dataset()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    model = ReidentificationModel(EMBEDDING_SIZE).to(device)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
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
                       optimizer, epoch, logger, LOG_INTERVAL)

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
