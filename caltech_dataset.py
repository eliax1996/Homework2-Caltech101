from torchvision.datasets import VisionDataset
import re
import os
from PIL import Image
import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    images = []

    def __init__(self, root,transform=None,split="train", target_transform=None): #
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        if split != "train" and split != "test":
            print("error: split must be or train or test")
            sys.exit(1)

        self.split = "Homework2_Caltech101/" + str(split) + ".txt"

        with open(self.split, 'r') as f:
            line = f.readline()

            for line in f:
                if re.match('.*BACKGROUND_Google.*', line):
                    continue

                data = ["./Homework2_Caltech101/101_ObjectCategories/" + line.strip()]
                data.append(line.split("/")[0])

                try:
                    image_loaded = pil_loader(data[0])
                    data[0] = image_loaded
                except:
                    print("ram finisced")

                self.images.append(data)

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
        #if self.transform is not None:
        #    image = self.transform(image)

        image,label = self.images[index]

        if not isinstance(image,Image.Image):
            image = pil_loader(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.images)
        return length
