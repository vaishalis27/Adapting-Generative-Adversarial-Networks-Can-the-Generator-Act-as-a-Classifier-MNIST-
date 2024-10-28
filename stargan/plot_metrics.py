import pickle
import matplotlib.pyplot as plt

# Load the metrics
with open('metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(metrics['g_gan_losses'], label='Generator GAN Loss')
plt.plot(metrics['g_clf_losses'], label='Generator Classification Loss')
plt.plot(metrics['g_rec_losses'], label='Generator Reconstruction Loss')
plt.plot(metrics['d_gan_losses'], label='Discriminator GAN Loss')
plt.plot(metrics['d_clf_losses'], label='Discriminator Classification Loss')
plt.legend()
plt.title('Losses Over Epochs')
plt.savefig('losses_plot.png')
plt.show()

# Plot the F1 Scores
plt.figure(figsize=(10, 5))
plt.plot(metrics['f1_scores_gen'], label='Generator F1 Score')
plt.plot(metrics['f1_scores_dis'], label='Discriminator F1 Score')
plt.legend()
plt.title('F1 Scores Over Epochs')
plt.savefig('f1_scores_plot.png')
plt.show()
