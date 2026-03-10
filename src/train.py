import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


############################################
# EDGE-AWARE GRADIENT LOSS
############################################

def gradient_loss(pred, target):

    pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])

    target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
    target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])

    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

def tv_loss(x):
    return torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))



############################################
# NEW: FEATURE COMPACTNESS LOSS
############################################

def compactness_loss(latent):

    dx = latent[:, :, :, 1:] - latent[:, :, :, :-1]
    dy = latent[:, :, 1:, :] - latent[:, :, :-1, :]

    loss = dx.pow(2).mean() + dy.pow(2).mean()

    return loss


############################################
# NEW: MULTI-SCALE RECONSTRUCTION LOSS
############################################

def multiscale_loss(pred, target):

    loss1 = F.l1_loss(pred, target)

    pred_half = F.interpolate(pred, scale_factor=0.5, mode='bilinear', align_corners=False)
    target_half = F.interpolate(target, scale_factor=0.5, mode='bilinear', align_corners=False)

    loss2 = F.l1_loss(pred_half, target_half)

    pred_quarter = F.interpolate(pred, scale_factor=0.25, mode='bilinear', align_corners=False)
    target_quarter = F.interpolate(target, scale_factor=0.25, mode='bilinear', align_corners=False)

    loss3 = F.l1_loss(pred_quarter, target_quarter)

    return loss1 + 0.5 * loss2 + 0.25 * loss3


############################################
# TRAIN AUTOENCODER (FAST VERSION)
############################################

def train_autoencoder(model, dataloader, epochs=100, device="cpu"):

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best_loss = float("inf")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    print("Training on:", device)

    for epoch in range(epochs):

        model.train()
        epoch_loss = 0

        for images in dataloader:

            images = images.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):

                    outputs, latent = model(images)

                    recon_loss = multiscale_loss(outputs, images)
                    edge_loss = gradient_loss(outputs, images)
                    compact_loss = compactness_loss(latent)

                    # ONLY NEW LINE
                    tv = tv_loss(outputs)

                    loss = (
                        recon_loss
                        + 0.6 * edge_loss
                        + 0.12 * compact_loss
                        + 0.05 * tv   # ONLY CHANGE
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                outputs, latent = model(images)

                recon_loss = multiscale_loss(outputs, images)
                edge_loss = gradient_loss(outputs, images)
                compact_loss = compactness_loss(latent)

                # ONLY NEW LINE
                tv = tv_loss(outputs)

                loss = (
                    recon_loss
                    + 0.6 * edge_loss
                    + 0.12 * compact_loss
                    + 0.05 * tv   # ONLY CHANGE
                )

                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "best_autoencoder.pth")
            print("Best model saved!")

    return model


############################################
# FAST LATENT EXTRACTION
############################################

def get_latent_features(model, dataloader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    features = []
    latents_all = []

    latent_h = None
    latent_w = None

    with torch.no_grad():

        for images in tqdm(dataloader, desc="Extracting latents"):

            images = images.to(device, non_blocking=True)

            _, latent = model(images)

            ###################################
            # LATENT SMOOTHING
            ###################################
            latent = F.avg_pool2d(
                latent,
                kernel_size=3,
                stride=1,
                padding=1
            )

            latents_all.append(latent.cpu())

            ###################################
            # BCHW → BHWC
            ###################################
            latent = latent.permute(0, 2, 3, 1).contiguous()

            B, H, W, C = latent.shape

            if latent_h is None:
                latent_h = H
                latent_w = W

            latent_pixels = latent.reshape(B * H * W, C).cpu().numpy()

            ###################################
            # POSITIONAL COORDS
            ###################################
            ys = np.linspace(0, 1, H)
            xs = np.linspace(0, 1, W)

            yy, xx = np.meshgrid(ys, xs, indexing='ij')

            coords = np.stack([yy, xx], axis=-1).reshape(-1, 2)
            coords = np.tile(coords, (B, 1))

            coords[:,0] *= 0.55
            coords[:,1] *= 0.10

            pixels = np.concatenate([latent_pixels, coords], axis=1)

            features.append(pixels)

    features = np.concatenate(features, axis=0)

    return features, latent_h, latent_w, latents_all
