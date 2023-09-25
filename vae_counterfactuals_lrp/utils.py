import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F



class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
    return torch.sum(kl_loss) / kl_loss.shape[0]

def loss_function_vae(recon_x, x, mu, log_var, beta=1):
    recon_loss = F.mse_loss(
        recon_x.reshape(x.shape[0], -1),
        x.reshape(x.shape[0], -1),
        reduction="none",
    ).sum(dim=-1)

    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

    return (recon_loss + beta*KLD).mean(dim=0)

def save_side_by_side(original, reconstructed, filename='comparison.png'):
    """
    Save two images side by side with labels.

    Parameters:
    - original: array of shape (3, 224, 224) representing the original image
    - reconstructed: array of shape (3, 224, 224) representing the reconstructed image
    - filename: path to save the output image
    """

    # Transpose the images to the shape 224x224x3
    if original.shape[2] != 3:
        original_img = np.transpose(original, (1, 2, 0))
    else:
        original_img = original

    if reconstructed.shape[2] != 3:
        reconstructed_img = np.transpose(reconstructed, (1, 2, 0))
    else:
        reconstructed_img = reconstructed

    # Normalize the images if necessary (assumes max value might be 255)
    #if original_img.max() > 1.0 or reconstructed_img.max() > 1.0:
    #    original_img = original_img / 255.0
    #    reconstructed_img = reconstructed_img / 255.0

    # Ensure the data is within [0, 1] range
    original_img = np.clip(original_img, 0, 1)
    reconstructed_img = np.clip(reconstructed_img, 0, 1)

    # Create the side-by-side plot
    fig, axarr = plt.subplots(1, 2, figsize=(10, 5))

    axarr[0].imshow(original_img)
    axarr[0].axis('off')
    axarr[0].set_title('Original')

    axarr[1].imshow(reconstructed_img)
    axarr[1].axis('off')
    axarr[1].set_title('Reconstructed')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5)
    plt.close()