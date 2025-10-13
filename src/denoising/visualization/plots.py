import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, psnr_vals, ssim_vals, save_path=None):
    """
    Plot and optionally save training/validation losses and PSNR/SSIM metrics.

    Args:
        train_losses (list): Training loss per epoch
        val_losses (list): Validation loss per epoch
        psnr_vals (list): PSNR per epoch
        ssim_vals (list): SSIM per epoch
        save_path (str): Optional path to save the plot
    """
    plt.figure(figsize=(10, 4))

    #loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    #PSNR and SSIM curves
    plt.subplot(1, 2, 2)
    plt.plot(psnr_vals, label='PSNR')
    plt.plot(ssim_vals, label='SSIM')
    plt.title('PSNR / SSIM over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
