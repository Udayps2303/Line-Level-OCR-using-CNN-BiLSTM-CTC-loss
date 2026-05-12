import os
import cv2
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────
MAPPING_FILE = "output/map_sorted.txt"
BASE_DIR = "./output"   # change if needed
WIDTH_BUCKET_SIZE = 100
HEIGHT_BUCKET_SIZE = 50

OUTPUT_FILE = "image_size.txt"
os.makedirs("output", exist_ok=True)

# ── Stats ─────────────────────────────────────────────────
total_images = 0

min_height = float("inf")
max_height = 0
total_height = 0

min_width = float("inf")
max_width = 0
total_width = 0

# ── Distributions ─────────────────────────────────────────
width_buckets = defaultdict(int)
height_buckets = defaultdict(int)

# ── Processing ────────────────────────────────────────────
with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue

        rel_path, _ = parts
        img_path = os.path.join(BASE_DIR, rel_path)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # ── Update stats ─────────────────────────────────
        total_images += 1

        total_height += h
        total_width += w

        min_height = min(min_height, h)
        max_height = max(max_height, h)

        min_width = min(min_width, w)
        max_width = max(max_width, w)

        # ── Width bucket (100 px) ────────────────────────
        w_start = (w // WIDTH_BUCKET_SIZE) * WIDTH_BUCKET_SIZE
        w_end = w_start + WIDTH_BUCKET_SIZE
        width_key = f"{w_start}-{w_end}"
        width_buckets[width_key] += 1

        # ── Height bucket (50 px) ────────────────────────
        h_start = (h // HEIGHT_BUCKET_SIZE) * HEIGHT_BUCKET_SIZE
        h_end = h_start + HEIGHT_BUCKET_SIZE
        height_key = f"{h_start}-{h_end}"
        height_buckets[height_key] += 1

# ── Final calculations ───────────────────────────────────
avg_height = total_height / total_images if total_images else 0
avg_width  = total_width / total_images if total_images else 0

# ── Sorting helpers ──────────────────────────────────────
def bucket_sort_key(bucket):
    return int(bucket.split("-")[0])

sorted_width = sorted(width_buckets.items(), key=lambda x: bucket_sort_key(x[0]))
sorted_height = sorted(height_buckets.items(), key=lambda x: bucket_sort_key(x[0]))

# ── Write to TXT ─────────────────────────────────────────
with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    out.write("=========== IMAGE SIZE STATISTICS ===========\n\n")

    out.write(f"Total Images: {total_images}\n\n")

    out.write("----- HEIGHT STATS -----\n")
    out.write(f"Min Height: {min_height}\n")
    out.write(f"Max Height: {max_height}\n")
    out.write(f"Avg Height: {avg_height:.2f}\n\n")

    out.write("----- WIDTH STATS -----\n")
    out.write(f"Min Width: {min_width}\n")
    out.write(f"Max Width: {max_width}\n")
    out.write(f"Avg Width: {avg_width:.2f}\n\n")

    out.write("=========== WIDTH DISTRIBUTION (100 px) ===========\n")
    for bucket, count in sorted_width:
        out.write(f"{bucket} px : {count} images\n")

    out.write("\n=========== HEIGHT DISTRIBUTION (50 px) ===========\n")
    for bucket, count in sorted_height:
        out.write(f"{bucket} px : {count} images\n")

# ── Console Output ───────────────────────────────────────
print("\n✅ Image size analysis completed!")
print(f"📄 Report saved at → {OUTPUT_FILE}")

print("\n--- SUMMARY ---")
print(f"Total Images: {total_images}")
print(f"Height → Min: {min_height}, Max: {max_height}, Avg: {avg_height:.2f}")
print(f"Width  → Min: {min_width}, Max: {max_width}, Avg: {avg_width:.2f}")