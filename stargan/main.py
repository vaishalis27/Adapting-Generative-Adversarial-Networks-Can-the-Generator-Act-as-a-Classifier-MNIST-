import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from model import Discriminator, Generator
from data_loader import get_loader
from utils import generate_imgs, gradient_penalty

# Configuration
EPOCHS = 200  # Adjust based on your requirement
BATCH_SIZE = 128
IMGS_TO_DISPLAY = 25
LOAD_MODEL = False

IMAGE_SIZE = 32
NUM_DOMAINS = 10  # For MNIST, 10 classes (digits 0-9)

N_CRITIC = 5
GRADIENT_PENALTY = 10 

# Directories for storing model and output samples
model_path = './model'
samples_path = './samples'
db_path = './data'

# Ensure directories exist
os.makedirs(model_path, exist_ok=True)
os.makedirs(samples_path, exist_ok=True)
os.makedirs(db_path, exist_ok=True)

# Initialize models, optimizers, and loss function
gen = Generator(num_domains=NUM_DOMAINS, image_size=IMAGE_SIZE)
dis = Discriminator(num_domains=NUM_DOMAINS, image_size=IMAGE_SIZE)

if LOAD_MODEL:
    gen.load_state_dict(torch.load(os.path.join(model_path, 'gen.pkl')))
    dis.load_state_dict(torch.load(os.path.join(model_path, 'dis.pkl')))

g_opt = optim.Adam(gen.parameters(), lr=0.0001, betas=(0.5, 0.999))
d_opt = optim.Adam(dis.parameters(), lr=0.0001, betas=(0.5, 0.999))

ce = nn.CrossEntropyLoss()

# Data loader
ds_loader = get_loader(db_path, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
iters_per_epoch = len(ds_loader)

# Fix images for visualization
img_fixed = next(iter(ds_loader))[0][:IMGS_TO_DISPLAY]

# GPU Compatibility
is_cuda = torch.cuda.is_available()
if is_cuda:
    gen, dis = gen.cuda(), dis.cuda()
    img_fixed = img_fixed.cuda()

# Initialize losses to avoid NameError
last_g_gan_loss = torch.tensor(0.0)
last_g_clf_loss = torch.tensor(0.0)
last_g_rec_loss = torch.tensor(0.0)

# Initialize storage for metrics
metrics = {
    'g_gan_losses': [],
    'g_clf_losses': [],
    'g_rec_losses': [],
    'd_gan_losses': [],
    'd_clf_losses': [],
    'f1_scores_gen': [],
    'f1_scores_dis': []
}

# Training loop with F1 score calculation and confusion matrix
for epoch in range(EPOCHS):
    gen.train()
    dis.train()

    all_true_labels_dis = []
    all_pred_labels_dis = []
    all_true_labels_gen = []
    all_pred_labels_gen = []

    total_iter = 0

    for i, data in enumerate(ds_loader):
        total_iter += 1
        real, dm = data
        
        if is_cuda:
            real, dm = real.cuda(), dm.cuda()

        target_dm = dm[torch.randperm(dm.size(0))]

        # Fake Images
        fake = gen(real, target_dm)

        # Training discriminator
        real_gan_out, real_cls_out = dis(real)
        fake_gan_out, fake_cls_out = dis(fake.detach())

        d_gan_loss = -(real_gan_out.mean() - fake_gan_out.mean()) + gradient_penalty(real, fake, dis, is_cuda) * GRADIENT_PENALTY
        d_clf_loss = ce(real_cls_out, dm)

        d_opt.zero_grad()
        d_loss = d_gan_loss + d_clf_loss
        d_loss.backward()
        d_opt.step()

        # Collect predictions and labels for F1 score and confusion matrix (Discriminator)
        _, pred_labels_dis = torch.max(real_cls_out, 1)
        all_true_labels_dis.extend(dm.cpu().numpy())
        all_pred_labels_dis.extend(pred_labels_dis.cpu().numpy())

        # Training Generator
        if total_iter % N_CRITIC == 0:
            fake = gen(real, target_dm)
            fake_gan_out, fake_cls_out = dis(fake)

            last_g_gan_loss = -fake_gan_out.mean()
            last_g_clf_loss = ce(fake_cls_out, target_dm)
            last_g_rec_loss = (real - gen(fake, dm)).abs().mean()

            g_opt.zero_grad()
            g_loss = last_g_gan_loss + last_g_clf_loss + last_g_rec_loss
            g_loss.backward()
            g_opt.step()

            # Collect predictions and labels for F1 score calculation (Generator)
            _, pred_labels_gen = torch.max(fake_cls_out, 1)
            all_true_labels_gen.extend(target_dm.cpu().numpy())
            all_pred_labels_gen.extend(pred_labels_gen.cpu().numpy())

        if i % 50 == 0:
            print("Epoch: " + str(epoch + 1) + "/" + str(EPOCHS)
                  + " iter: " + str(i+1) + "/" + str(iters_per_epoch)
                  + " total_iters: " + str(total_iter)
                  + "\td_gan_loss:" + str(round(d_gan_loss.item(), 4))
                  + "\td_clf_loss:" + str(round(d_clf_loss.item(), 4))
                  + "\tg_gan_loss:" + str(round(last_g_gan_loss.item(), 4))
                  + "\tg_clf_loss:" + str(round(last_g_clf_loss.item(), 4))
                  + "\tg_rec_loss:" + str(round(last_g_rec_loss.item(), 4)))

    # Calculate and store F1 scores for the epoch
    precision_dis = precision_score(all_true_labels_dis, all_pred_labels_dis, average='macro')
    recall_dis = recall_score(all_true_labels_dis, all_pred_labels_dis, average='macro')
    f1_dis = f1_score(all_true_labels_dis, all_pred_labels_dis, average='macro')
    metrics['f1_scores_dis'].append(f1_dis)

    precision_gen = precision_score(all_true_labels_gen, all_pred_labels_gen, average='macro')
    recall_gen = recall_score(all_true_labels_gen, all_pred_labels_gen, average='macro')
    f1_gen = f1_score(all_true_labels_gen, all_pred_labels_gen, average='macro')
    metrics['f1_scores_gen'].append(f1_gen)

    metrics['g_gan_losses'].append(last_g_gan_loss.item())
    metrics['g_clf_losses'].append(last_g_clf_loss.item())
    metrics['g_rec_losses'].append(last_g_rec_loss.item())
    metrics['d_gan_losses'].append(d_gan_loss.item())
    metrics['d_clf_losses'].append(d_clf_loss.item())

    # Generate and save confusion matrix for discriminator every 10 epochs
    if (epoch + 1) % 10 == 0:  # Plot every 10 epochs
        cm = confusion_matrix(all_true_labels_dis, all_pred_labels_dis)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f'Confusion Matrix at Epoch {epoch+1}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(f'confusion_matrix_epoch_{epoch+1}.png')  # Save the figure
        plt.close()

    # Save models and generate images every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(gen.state_dict(), os.path.join(model_path, f'gen_epoch_{epoch+1}.pkl'))
        torch.save(dis.state_dict(), os.path.join(model_path, f'dis_epoch_{epoch+1}.pkl'))
        generate_imgs(img_fixed, NUM_DOMAINS, gen, samples_path=samples_path, step=epoch+1, is_cuda=is_cuda)

# Save the metrics at the end of training
with open('metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

# Save the losses plot
plt.figure(figsize=(10, 5))
plt.plot(metrics['g_gan_losses'], label='Generator GAN Loss')
plt.plot(metrics['g_clf_losses'], label='Generator Classification Loss')
plt.plot(metrics['g_rec_losses'], label='Generator Reconstruction Loss')
plt.plot(metrics['d_gan_losses'], label='Discriminator GAN Loss')
plt.plot(metrics['d_clf_losses'], label='Discriminator Classification Loss')
plt.legend()
plt.title('Losses Over Epochs')
plt.savefig('losses_over_epochs.png')  # Save the figure
plt.close()  # Close the figure to free memory

# Save the F1 Scores plot
plt.figure(figsize=(10, 5))
plt.plot(metrics['f1_scores_gen'], label='Generator F1 Score')
plt.plot(metrics['f1_scores_dis'], label='Discriminator F1 Score')
plt.legend()
plt.title('F1 Scores Over Epochs')
plt.savefig('f1_scores_over_epochs.png')  # Save the figure
plt.close()  # Close the figure to free memory
