import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2


###################################################
# STRONGER RETINA MASK (Otsu + smoothing)
###################################################

def get_retina_mask(image):
    """
    Robust retina detection using vertical profile + Otsu threshold.
    Much more stable across brightness variations.
    """

    img = image.squeeze()

    # despeckle
    img = gaussian_filter(img, sigma=2)

    # normalize for cv2
    img_8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, thresh = cv2.threshold(
        img_8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask = thresh > 0

    return mask.astype(np.uint8)


###################################################
# BETTER CONV BLOCK
# (GroupNorm + Dropout)
###################################################

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(groups, out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(groups, out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


###################################################
# RESEARCH-GRADE AUTOENCODER (U-Net Style)
###################################################

class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        # ---------- ENCODER ----------

        self.enc1 = ConvBlock(1, 32)      
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64)    
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(64, 128)   
        self.pool3 = nn.MaxPool2d(2)

        #  Wider latent improves clustering massively
        self.enc4 = ConvBlock(128, 256)

        # optional smoothing inside network
        #self.latent_smooth = nn.AvgPool2d(
            #kernel_size=3,
            #stride=1,
            #padding=1
        #)

        # ---------- DECODER ----------

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.refine = nn.Sequential(
        nn.Conv2d(32, 32, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 1, 1))

    def forward(self, x):

        # ----- Encoder -----

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        latent = self.enc4(p3)

        ##################################
        #  LATENT SMOOTHING INSIDE MODEL
        ##################################
        #latent = self.latent_smooth(latent)

        # ----- Decoder -----

        d3 = self.up3(latent)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = torch.sigmoid(self.refine(d1))


        return out, latent
