import torch
from torch import optim
import matplotlib.pyplot as plt

from denoising.models import DnCNN
from denoising.losses import ResidualMSELoss
from denoising.metrics import psnr, ssim_metric
from denoising.visualization import plot_training_curves

def train_dncnn(model, train_loader, val_loader, device, epochs=50, lr=1e-1):
    criterion = ResidualMSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(1e-4/lr)**(1/epochs))

    train_losses, val_losses, psnr_vals, ssim_vals = [], [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()

            residual_pred = model(noisy)
            loss = criterion(residual_pred, noisy, clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        scheduler.step()

        #validation
        model.eval()
        val_loss, psnr_epoch, ssim_epoch = 0.0, 0.0, 0.0

        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                residual_pred = model(noisy) #R(y;Θ)
                loss = criterion(residual_pred, noisy, clean)
                val_loss += loss.item()

                denoised = (noisy - residual_pred).clamp(0.0, 1.0) #x̂ = y - R(y;Θ)
                psnr_epoch += psnr(denoised, clean).item()
                ssim_epoch += ssim_metric(denoised, clean).item()

        val_loss /= len(val_loader)
        psnr_epoch /= len(val_loader)
        ssim_epoch /= len(val_loader)

        val_losses.append(val_loss)
        psnr_vals.append(psnr_epoch)
        ssim_vals.append(ssim_epoch)

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f" Train Loss: {train_loss:.6f}")
        print(f" Val Loss: {val_loss:.6f}")
        print(f" PSNR: {psnr_epoch:.2f}")
        print(f" SSIM: {ssim_epoch:.3f}")
        print("-" * 40)

    plot_training_curves(train_losses, val_losses, psnr_vals, ssim_vals)

    return model