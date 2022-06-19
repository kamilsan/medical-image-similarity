import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import csv
import os
import random
from PIL import Image


class HAM1000Dataset(Dataset):
    full_metadata = []

    def __init__(self, dataset_directory, split):
        super(HAM1000Dataset, self).__init__()
        self.dataset_directory = dataset_directory

        if len(self.full_metadata) == 0:
            metadata_file = os.path.join(
                dataset_directory, 'HAM10000_metadata')

            self.dataset_metadata = []

            with open(metadata_file, mode='r') as metadata_file:
                metadata_csv = csv.reader(metadata_file)
                for line in list(metadata_csv)[1:]:
                    image_filename = line[1]
                    diagnosis = line[2]
                    self.full_metadata.append({
                        'image_name': image_filename,
                        'diagnosis': diagnosis
                    })

            random.shuffle(self.full_metadata)

        validation_split_start_idx = int(0.8 * len(self.full_metadata))
        test_split_start_idx = int(0.9 * len(self.full_metadata))

        if split == 'train':
            self.dataset_metadata = self.full_metadata[:validation_split_start_idx]
        elif split == 'val':
            self.dataset_metadata = self.full_metadata[validation_split_start_idx:test_split_start_idx]
        elif split == 'test':
            self.dataset_metadata = self.full_metadata[test_split_start_idx:]
        else:
            raise Exception('Unknown split type!')

        self.group_data()

    def group_data(self):
        all_diagnoses = set()

        for entry in self.dataset_metadata:
            diagnosis = entry['diagnosis']
            if not diagnosis in all_diagnoses:
                all_diagnoses.add(diagnosis)

        self.grouped_data = {}
        for diagnosis in all_diagnoses:
            images = []
            for entry in self.dataset_metadata:
                if entry['diagnosis'] == diagnosis:
                    images.append(entry['image_name'])
            self.grouped_data[diagnosis] = images

        self.diagnoses = list(all_diagnoses)

    def load_image(self, image_name):
        image_path = os.path.join(
            self.dataset_directory, 'images', f'{image_name}.jpg')
        image = Image.open(image_path)

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
        tensor = transform(image)

        return tensor

    def __len__(self):
        return len(self.dataset_metadata)

    def __getitem__(self, idx):
        anchor_class, negative_class = random.choices(self.diagnoses, k=2)

        anchor_image_name, positive_image_name = random.choices(
            self.grouped_data[anchor_class], k=2)
        negative_image_name = random.choice(self.grouped_data[negative_class])

        anchor_image = self.load_image(anchor_image_name)
        positive_image = self.load_image(positive_image_name)
        negative_image = self.load_image(negative_image_name)

        return anchor_image, positive_image, negative_image
