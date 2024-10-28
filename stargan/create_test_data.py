import os
import shutil
from torchvision import datasets

# Create test data directory structure
test_data_path = './test_data'
os.makedirs(test_data_path, exist_ok=True)

# Download MNIST test set
test_set = datasets.MNIST(root='./mnist_data', train=False, download=True)

# Save MNIST test images into class-specific folders
for idx, (img, label) in enumerate(test_set):
    label_dir = os.path.join(test_data_path, str(label))
    os.makedirs(label_dir, exist_ok=True)
    img.save(os.path.join(label_dir, f'test_{idx}.png'))
