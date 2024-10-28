import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os
from model import Discriminator, Generator
from data_loader import get_loader
from utils import generate_imgs, gradient_penalty

# Configuration
BATCH_SIZE = 128
IMGS_TO_DISPLAY = 10  # Number of images per class to display
IMAGE_SIZE = 32
NUM_DOMAINS = 10  # For MNIST, 10 classes (digits 0-9)

# Directories for storing model and output samples
model_path = './model'
samples_path = './samples'
db_path = './data'
test_data_path = './test_data'  # Path to your test dataset

# Ensure directories exist
os.makedirs(model_path, exist_ok=True)
os.makedirs(samples_path, exist_ok=True)
os.makedirs(db_path, exist_ok=True)

# Initialize models
gen = Generator(num_domains=NUM_DOMAINS, image_size=IMAGE_SIZE)
dis = Discriminator(num_domains=NUM_DOMAINS, image_size=IMAGE_SIZE)

# Load trained models
gen.load_state_dict(torch.load(os.path.join(model_path, 'gen_epoch_155.pkl')))
dis.load_state_dict(torch.load(os.path.join(model_path, 'dis_epoch_155.pkl')))

# GPU Compatibility
is_cuda = torch.cuda.is_available()
if is_cuda:
    gen, dis = gen.cuda(), dis.cuda()

# Switch models to evaluation mode
gen.eval()
dis.eval()

# Test Data Loader
test_loader = get_loader(test_data_path, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)

# Evaluate on the Test Set
all_true_labels_test = []
all_pred_labels_test = []

for i, data in enumerate(test_loader):
    real, dm = data

    if is_cuda:
        real, dm = real.cuda(), dm.cuda()

    target_dm = dm[torch.randperm(dm.size(0))]
    
    # Generate fake images using the generator
    fake = gen(real, target_dm)
    
    # Get predictions from the discriminator
    _, fake_cls_out = dis(fake)
    _, pred_labels_test = torch.max(fake_cls_out, 1)
    
    all_true_labels_test.extend(target_dm.cpu().numpy())
    all_pred_labels_test.extend(pred_labels_test.cpu().numpy())

# Calculate and print test precision, recall, F1 score, and accuracy
precision_test = precision_score(all_true_labels_test, all_pred_labels_test, average='macro')
recall_test = recall_score(all_true_labels_test, all_pred_labels_test, average='macro')
f1_test = f1_score(all_true_labels_test, all_pred_labels_test, average='macro')
accuracy_test = accuracy_score(all_true_labels_test, all_pred_labels_test)

print(f"Test Precision: {precision_test:.4f}")
print(f"Test Recall: {recall_test:.4f}")
print(f"Test F1 Score: {f1_test:.4f}")
print(f"Test Accuracy: {accuracy_test:.4f}")

# Visual Inspection of Generated Images

# Use some real images from the test set to generate new images
real_images, _ = next(iter(test_loader))
real_images = real_images[:IMGS_TO_DISPLAY]  # Select the first few images

if is_cuda:
    real_images = real_images.cuda()

# Generating images for each domain (class)
generated_images = []
for i in range(NUM_DOMAINS):
    labels = torch.full((IMGS_TO_DISPLAY,), i, dtype=torch.long)
    if is_cuda:
        labels = labels.cuda()
    gen_images = gen(real_images, labels)
    generated_images.append(gen_images)

# Concatenate all generated images row-wise for display
generated_images = torch.cat([img for img in generated_images], dim=0)
generated_images = (generated_images + 1) / 2  # Assuming images were normalized to [-1, 1]

# Ensure the images are displayed in a grid with one class per row
nrows = NUM_DOMAINS
ncols = IMGS_TO_DISPLAY
grid = vutils.make_grid(generated_images, nrow=ncols, padding=2, normalize=True)

# Display the generated images
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(grid.permute(1, 2, 0).cpu())
plt.show()

# Save the generated images
vutils.save_image(grid, './generated_samples.png', normalize=True)
