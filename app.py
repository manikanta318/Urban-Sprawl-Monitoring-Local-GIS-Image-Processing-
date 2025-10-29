# =====================================================
# 🌆 Urban Sprawl Monitoring — Universal Version
# Handles GeoTIFF & standard images | Auto-installs packages
# =====================================================

import importlib, subprocess, sys

def install_if_missing(packages):
    """Auto-install required Python packages if not already installed."""
    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError:
            print(f"⚙️ Installing missing package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

# === Step 1: Check & Install Dependencies ===
required_packages = [
    "rasterio", "numpy", "matplotlib", "opencv-python"
]
install_if_missing(required_packages)

# === Step 2: Import Libraries ===
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings
import os

# === Step 3: Silence non-critical warnings ===
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# === Step 4: Helper — NDBI Calculation ===
def calculate_ndbi(nir, swir):
    """Normalized Difference Built-up Index"""
    ndbi = (swir - nir) / (swir + nir + 1e-10)  # avoid division by zero
    ndbi[np.isinf(ndbi)] = np.nan
    return ndbi

# === Step 5: Smart Image Loader (GeoTIFF or Standard Image) ===
def load_image_auto(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.tif', '.tiff']:
        src = rasterio.open(path)
        bands = src.count
        print(f"🛰️ Loaded GeoTIFF with {bands} bands")

        if bands >= 6:
            nir = src.read(5).astype('float32')
            swir = src.read(6).astype('float32')
            print("✅ Using NIR=Band5 & SWIR=Band6")
        elif bands >= 3:
            red = src.read(1).astype('float32')
            green = src.read(2).astype('float32')
            blue = src.read(3).astype('float32')
            nir, swir = red, blue
            print("⚠️ Only RGB bands found — simulating NIR=Red, SWIR=Blue")
        else:
            print("❌ Too few bands — generating synthetic data for testing.")
            nir = np.random.rand(512, 512)
            swir = np.random.rand(512, 512)
    else:
        print(f"🖼️ Loaded standard image: {path}")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
        red, green, blue = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        nir, swir = red, blue
        print("⚠️ Non-GIS image — simulating NIR=Red & SWIR=Blue")

    return nir, swir

# === Step 6: File Paths (update if needed) ===
image_2010_path = "/content/landsat_2010.webp"
image_2020_path = "/content/landsat_2020.webp"

# === Step 7: Load Images ===
nir_2010, swir_2010 = load_image_auto(image_2010_path)
nir_2020, swir_2020 = load_image_auto(image_2020_path)

# === Step 8: Compute NDBI ===
ndbi_2010 = calculate_ndbi(nir_2010, swir_2010)
ndbi_2020 = calculate_ndbi(nir_2020, swir_2020)

# === Step 9: Classify Built-up ===
urban_2010 = np.where(ndbi_2010 > 0, 1, 0)
urban_2020 = np.where(ndbi_2020 > 0, 1, 0)

# === Step 10: Detect Urban Change ===
change_map = urban_2020 - urban_2010  # 1 = new urban, -1 = lost urban

# === Step 11: Visualization ===
plt.figure(figsize=(16, 8))
plt.subplot(2, 3, 1)
plt.imshow(ndbi_2010, cmap='terrain')
plt.title("NDBI 2010")

plt.subplot(2, 3, 2)
plt.imshow(ndbi_2020, cmap='terrain')
plt.title("NDBI 2020")

plt.subplot(2, 3, 3)
plt.imshow(change_map, cmap='bwr', vmin=-1, vmax=1)
plt.title("Urban Change (Blue=Lost, Red=New)")

plt.subplot(2, 3, 4)
plt.imshow(urban_2010, cmap='gray')
plt.title("Urban Mask 2010")

plt.subplot(2, 3, 5)
plt.imshow(urban_2020, cmap='gray')
plt.title("Urban Mask 2020")

plt.subplot(2, 3, 6)
plt.imshow(change_map, cmap='bwr', vmin=-1, vmax=1)
plt.title("Urban Change Summary")

plt.tight_layout()
plt.show()

print("✅ Urban sprawl analysis complete.")
