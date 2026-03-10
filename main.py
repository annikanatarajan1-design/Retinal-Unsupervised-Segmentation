from src.dataset import OCTDataset
from src.train import train_autoencoder, get_latent_features
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.ndimage import median_filter, gaussian_filter
from src.utils import Autoencoder, get_retina_mask

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import hdbscan
import matplotlib


def main():

    ####################################################
    # DEVICE
    ####################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ####################################################
    # DATA
    ####################################################

    dataset = OCTDataset("data/images", img_size=128)

    dataloader = DataLoader(
        dataset,
        batch_size=6,
        shuffle=False,
        num_workers=0
    )

    ####################################################
    # MODEL
    ####################################################

    model = Autoencoder().to(device)

    ####################################################
    # TRAIN ONLY IF MODEL DOES NOT EXIST
    ####################################################

    if not os.path.exists("models/best_autoencoder.pth"):

        print("\nNo trained model found.")
        print("Training Autoencoder for 100 epochs...\n")

        model = train_autoencoder(
            model,
            dataloader,
            epochs=100,
            device=device
        )

    else:

        print("Loading trained autoencoder...\n")

        model.load_state_dict(
            torch.load("models/best_autoencoder.pth", map_location=device)
        )

    model.eval()

    ####################################################
    # FEATURE EXTRACTION
    ####################################################

    print("Extracting latent features...\n")

    features, latent_h, latent_w, latents_all = get_latent_features(
        model,
        dataloader
    )

    pixels_per_image = latent_h * latent_w

    ####################################################
    # FASTER SOBEL (vectorized)
    ####################################################

    print("Adding Sobel features...\n")

    edge_features = []

    for latent_batch in latents_all:

        latent_np = latent_batch.numpy()   # (B, C, H, W)

        dx = np.diff(latent_np, axis=3, prepend=latent_np[:, :, :, :1])
        dy = np.diff(latent_np, axis=2, prepend=latent_np[:, :, :1, :])

        grad_mag = np.sqrt(dx**2 + dy**2)

        grad_mag = grad_mag.mean(axis=1)   # (B, H, W)

        grad_mag = grad_mag.reshape(-1, 1)

        edge_features.append(grad_mag)

    edge_features = np.vstack(edge_features)
    edge_features *= 0.25
    features = np.concatenate([features, edge_features], axis=1)
    
    ####################################################
    # STANDARDIZE + PCA
    ####################################################

    #scaler = StandardScaler()
    #features = scaler.fit_transform(features)

    pca = PCA(
        n_components=32,
        whiten=True,
        random_state=42
    )

    features = pca.fit_transform(features)

    ####################################################
    # HDBSCAN (ANATOMY-SAFE CLUSTERING)
    ####################################################

    print("Running HDBSCAN...\n")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=800,
        min_samples=250,
        metric='euclidean',
        cluster_selection_epsilon=0.02
    )

    labels = clusterer.fit_predict(features)
    print("Clusters found:", len(np.unique(labels)))

    num_images = len(dataset)

    labels = labels.reshape(num_images, latent_h, latent_w)

    # convert noise (-1) → background (0)
    labels[labels == -1] = 0

    labels = labels.reshape(-1)

    ####################################################
    # VISUALIZATION
    ####################################################

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))

    cmap = matplotlib.colormaps['nipy_spectral']

    for i, ax in enumerate(axes.flatten()):

        if i >= num_images:
            ax.axis("off")
            continue

        start = i * pixels_per_image
        end = (i + 1) * pixels_per_image

        segmented_image = labels[start:end].reshape(latent_h, latent_w)

        ################################################
        # VERTICAL ORDERING
        ################################################

        unique_clusters = np.unique(segmented_image)

        cluster_means = []

        for c in unique_clusters:
            ys = np.where(segmented_image == c)[0]
            mean_y = np.mean(ys) if len(ys) > 0 else 1e9
            cluster_means.append((c, mean_y))

        # sort from top → bottom
        cluster_means.sort(key=lambda x: x[1])

        remap = {old: new for new, (old, _) in enumerate(cluster_means)}

        for old, new in remap.items():
            segmented_image[segmented_image == old] = new

        ################################################
        # LOAD OCT IMAGE
        ################################################

        oct_img = dataset[i].squeeze(0).numpy()
        H, W = oct_img.shape

        ################################################
        # UPSCALE
        ################################################

        segmented_image = cv2.resize(
            segmented_image.astype(np.float32),
            (W, H),
            interpolation=cv2.INTER_NEAREST
        )

        segmented_image = median_filter(segmented_image, size=(3, 3))

        #segmented_image = gaussian_filter(segmented_image, sigma=(0.6, 1.2))

        ################################################
        # RETINA MASK
        ################################################

        mask = get_retina_mask(oct_img)
        segmented_image[mask == 0] = 0

        ################################################
        # PLOT
        ################################################

        ax.imshow(oct_img, cmap='nipy_spectral')

        ax.imshow(
            segmented_image,
            cmap=cmap,
            alpha=0.55,
            interpolation='nearest'
        )

        ax.set_title("Unsupervised", fontsize=10)
        ax.axis("off")

    ####################################################
    # SAVE FIGURE
    ####################################################

    plt.savefig(
        "outputs/retinal_segmentation.png",
        dpi=600,
        bbox_inches='tight',
        pad_inches=0.3
    )

    plt.close()

    print("\n Segmentation saved as retinal_segmentation.png\n")


if __name__ == "__main__":
    main()
