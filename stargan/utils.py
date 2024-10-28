import torch
import torchvision.utils as vutils
import math
import os
from torch import autograd
import matplotlib.pyplot as plt
import time


def generate_imgs(img, n_dms, gen, samples_path=None, step=0, is_cuda=True):
    gen.eval()
    m = img.shape[0]

    lbl = torch.arange(start=-1, end=n_dms)
    lbl = lbl.expand(m, n_dms + 1).reshape([-1])

    if is_cuda:
        lbl = lbl.cuda()
    img_ = torch.repeat_interleave(img, n_dms + 1, dim=0)

    real_idx = torch.arange(start=0, end=m * (n_dms + 1), step=n_dms + 1)
    lbl[real_idx] = 0

    display_imgs = gen(img_, lbl)
    display_imgs[real_idx] = img

    display_imgs_ = vutils.make_grid(display_imgs, normalize=True, nrow=n_dms + 1, padding=2, pad_value=1)

    # Show the image using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(display_imgs_.cpu().permute(1, 2, 0))  # Permute to change (C, H, W) to (H, W, C)
    plt.axis('off')  # Hide axes
    plt.title(f'Generated images at step {step}')
    plt.show()  # Show the plot and block execution until closed

    # Optionally save the image as well
    if samples_path:
        vutils.save_image(display_imgs_, os.path.join(samples_path, f'sample_{step}.png'))

def gradient_penalty(real, fake, critic, is_cuda=True):
	m = real.shape[0]
	epsilon = torch.rand(m, 1, 1, 1)
	if is_cuda:
		epsilon = epsilon.cuda()
	
	interpolated_img = epsilon * real + (1-epsilon) * fake
	interpolated_out, _ = critic(interpolated_img)

	grads = autograd.grad(outputs=interpolated_out, inputs=interpolated_img,
							   grad_outputs=torch.ones(interpolated_out.shape).cuda() if is_cuda else torch.ones(interpolated_out.shape),
							   create_graph=True, retain_graph=True)[0]
	grads = grads.reshape([m, -1])
	grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean() 
	return grad_penalty