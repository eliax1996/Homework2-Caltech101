from torchvision.datasets import VisionDataset
import re
import os
from PIL import Image

import os
import os.path
import sys


# print('file is correctly loaded')


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split="train", transform=None, target_transform=None):
        split = str(split)
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        if self.split != "train" and self.split != "test":
            print("error: split must be or train or test")
            sys.exit(1)

        self.split = str(os.path.dirname(os.path.abspath(__file__))) + str(self.split) + ".txt"

        '''
        with open(self.split, 'r') as f:
            line = f.readline()

            while line:
                if re.match('.*BACKGROUND_Google.*', line):
                    continue

                print(line)
                
        '''

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        #image, label = ...  # Provide a way to access image and label via index
        # Image should be a PIL Image
        # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        #length = ...  # Provide a way to get the length (number of elements) of the dataset
        return length
