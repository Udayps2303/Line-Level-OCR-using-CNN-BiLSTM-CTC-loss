import cv2
import numpy as np
import os
import random

# ---------- INPUT IMAGE ----------
img_path = "output/OdiaLineLevelDataSet/1_op/1_0001-916/1_0001-916_line_2.png"

# ---------- OUTPUT FOLDERS ----------
AFFINE_DIR = "affine_output"
VIGNETTE_DIR = "vignette_output"
SKEW_DIR = "skew_output"

ELASTIC_DIR = "elastic_output"
MORPH_DIR = "morph_output"
CJITTER_DIR = "colorjitter_output"

NOISE_DIR = "noise_output"
CORROSION_DIR = "corrosion_output"

for d in [AFFINE_DIR, VIGNETTE_DIR, SKEW_DIR,
          ELASTIC_DIR, MORPH_DIR, CJITTER_DIR,
          NOISE_DIR, CORROSION_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------- ORIGINAL FUNCTIONS ----------

def apply_affine(img):
    h, w = img.shape[:2]

    pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    pts2 = np.float32([
        [0, 0],
        [int(0.97 * (w - 1)), int(0.03 * h)],
        [int(0.03 * w), int(0.97 * (h - 1))]
    ])

    M = cv2.getAffineTransform(pts1, pts2)

    return cv2.warpAffine(
        img, M, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )


def apply_vignette(img):
    rows, cols = img.shape[:2]

    direction = np.random.choice(['lr', 'rl', 'tb', 'bt'])
    strength = np.random.uniform(0.6, 0.9)

    if direction == 'lr':
        gradient = np.tile(np.linspace(strength, 1.0, cols), (rows, 1))
    elif direction == 'rl':
        gradient = np.tile(np.linspace(1.0, strength, cols), (rows, 1))
    elif direction == 'tb':
        gradient = np.tile(np.linspace(strength, 1.0, rows), (cols, 1)).T
    else:
        gradient = np.tile(np.linspace(1.0, strength, rows), (cols, 1)).T

    vignette = img.astype(np.float32)
    for i in range(3):
        vignette[:, :, i] *= gradient

    return vignette.astype(np.uint8)


def apply_skew(img, angle=10):
    h, w = img.shape[:2]

    M = np.float32([
        [1, np.tan(np.radians(angle)), 0],
        [0, 1, 0]
    ])

    new_w = int(w + abs(h * np.tan(np.radians(angle))))

    return cv2.warpAffine(
        img, M, (new_w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

# ---------- NEW AUGMENTATIONS ----------

def apply_elastic(img, alpha=20, sigma=5):
    h, w = img.shape[:2]

    dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1),
                          (17, 17), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1),
                          (17, 17), sigma) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))

    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    return cv2.remap(img, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=(255, 255, 255))


def apply_morphology(img):
    op = random.choice(["erode", "dilate"])
    k = random.randint(1, 3)

    kernel = np.ones((k, k), np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if op == "erode":
        morphed = cv2.erode(gray, kernel)
    else:
        morphed = cv2.dilate(gray, kernel)

    return cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)


def apply_color_jitter(img):
    brightness = random.uniform(0.7, 1.3)
    contrast = random.uniform(0.7, 1.3)

    img = img.astype(np.float32)
    img = img * brightness

    mean = np.mean(img)
    img = (img - mean) * contrast + mean

    return np.clip(img, 0, 255).astype(np.uint8)

# ---------- NEW FILTERS YOU REQUESTED ----------

# 1. GAUSSIAN NOISE
def apply_gaussian_noise(img, mean=0, std=3):
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# 2. CORROSION EFFECT (Missing Ink)
def apply_corrosion(img):
    img = img.astype(np.float32)
    h, w = img.shape[:2]

    intensity = np.random.rand() ** 4.0
    prob = min(0.012, intensity)

    num_patches = int(prob * h * w)

    for _ in range(num_patches):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)

        patch_size = random.randint(1, int(10 * intensity) + 1)

        for i in range(-patch_size, patch_size):
            for j in range(-patch_size, patch_size):
                xi = x + i
                yj = y + j

                if 0 <= xi < w and 0 <= yj < h:
                    if random.random() < 0.7:  # irregular shape
                        dist = np.sqrt(i**2 + j**2)
                        falloff = max(0, 1 - dist / (patch_size + 1))

                        ink_loss = random.uniform(25, 500) * falloff
                        img[yj, xi] += ink_loss

    return np.clip(img, 0, 255).astype(np.uint8)


# ---------- PROCESS ----------

img = cv2.imread(img_path)

if img is None:
    print("Error: Image not found!")
    exit()

filename = os.path.basename(img_path)

# ORIGINAL
cv2.imwrite(os.path.join(AFFINE_DIR, filename), apply_affine(img))
cv2.imwrite(os.path.join(VIGNETTE_DIR, filename), apply_vignette(img))
cv2.imwrite(os.path.join(SKEW_DIR, filename), apply_skew(img))

# EXISTING NEW
cv2.imwrite(os.path.join(ELASTIC_DIR, filename), apply_elastic(img))
cv2.imwrite(os.path.join(MORPH_DIR, filename), apply_morphology(img))
cv2.imwrite(os.path.join(CJITTER_DIR, filename), apply_color_jitter(img))

# NEWLY ADDED
cv2.imwrite(os.path.join(NOISE_DIR, filename), apply_gaussian_noise(img))
cv2.imwrite(os.path.join(CORROSION_DIR, filename), apply_corrosion(img))

print("All preprocessing methods applied successfully!")