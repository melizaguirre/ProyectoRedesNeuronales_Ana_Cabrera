import numpy as np
import struct

class MnistDataset:
    def __init__(self):
        self.images = None
        self.labels = None

    def load(self, images_filename, labels_filename):
        # Load images
        with open(images_filename, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))

            if magic != 2051:
                raise ValueError(f"Invalid magic number {magic}. Expected 2051 for images.")

            buffer = f.read()

            data = np.frombuffer(buffer, dtype=np.uint8)

            self.images = data.reshape(num_images, rows * cols)

            self.images = self.images / 255.0

        # Load labels
        with open(labels_filename, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))

            if magic != 2049:
                raise ValueError(f"Invalid magic number {magic}. Expected 2049 for labels.")

            self.labels = np.frombuffer(f.read(), dtype=np.uint8)

            self.one_hot_labels = self.convert_to_one_hot(self.labels)

    def convert_to_one_hot(self, labels):
        one_hot_labels = np.zeros((len(labels), 10), dtype=np.float32)
        one_hot_labels[np.arange(len(labels)), labels] = 1
        return one_hot_labels
    