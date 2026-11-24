import numpy as np


def shades_of_gray(img, power=6, gamma=None):
    """Shades of Gray colour constancy (Finlayson & Trezzi, 2004), common in
    skin lesion work (Barata et al., 2015). img: HxWx3 uint8 RGB."""
    x = img.astype(np.float32)
    if gamma is not None:
        x = 255.0 * np.power(x / 255.0, 1.0 / gamma)
    illum = np.power(np.mean(np.power(x, power), axis=(0, 1)), 1.0 / power)
    illum = illum / (np.sqrt(np.sum(illum ** 2)) + 1e-6)
    x = x / (illum * np.sqrt(3) + 1e-6)
    return np.clip(x, 0, 255).astype(np.uint8)
