''' Code to create the final mapping file (map.txt) by performing OCR on line images and matching 
with GT text.'''
import os
import re
import cv2
import pytesseract
from rapidfuzz import fuzz
from multiprocessing import Pool, cpu_count

# CONFIG
BASE_DATASET = "output/OdiaLineLevelDataSet"
GT_PATH = "input/books/GT"
OUTPUT_FILE = "output/map.txt"

SIM_THRESHOLD = 70
MIN_TEXT_LEN = 5
OCR_LANG = "ori+eng"

NUM_WORKERS = max(1, cpu_count() - 1)


# ---------------------------
# CLEANING
# ---------------------------
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip().lower()


def is_noise(text):
    if len(text) < MIN_TEXT_LEN:
        return True
    if re.fullmatch(r'[\*\-\_=\.]+', text):
        return True
    return False


# ---------------------------
# OCR + MATCH (PARALLEL UNIT)
# ---------------------------
def process_single(args):
    img_path, rel_path, gt_lines = args

    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(
        gray,
        lang=OCR_LANG,
        config="--psm 7"
    )

    ocr_text = clean_text(text)

    # ❌ Noise
    if is_noise(ocr_text):
        return None

    # ❌ Multi-line skip
    if len(ocr_text.split()) > 25:
        return None

    # 🔍 FULL search (no pointer here)
    best_score = 0
    best_text = None

    for gt in gt_lines:
        score = fuzz.ratio(ocr_text, gt)
        if score > best_score:
            best_score = score
            best_text = gt

    # ❌ weak match
    if best_score < SIM_THRESHOLD:
        return None

    return (rel_path, best_text)


# ---------------------------
# LOAD GT
# ---------------------------
def load_book_gt(book_id):
    gt_file = os.path.join(GT_PATH, f"{book_id}.txt")

    lines = []
    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            cleaned = clean_text(line)
            if not is_noise(cleaned):
                lines.append(cleaned)

    return lines


# ---------------------------
# MAIN
# ---------------------------
def main():

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:

        for book_id in range(1, 44):

            print(f"\nProcessing Book {book_id} with {NUM_WORKERS} workers...")

            folder_path = os.path.join(BASE_DATASET, f"{book_id}_op")
            gt_lines = load_book_gt(book_id)

            tasks = []

            pages = sorted(os.listdir(folder_path))

            for page in pages:
                page_path = os.path.join(folder_path, page)

                if not os.path.isdir(page_path):
                    continue

                image_files = sorted([
                    f for f in os.listdir(page_path)
                    if "line_" in f and f.endswith(".png")
                ])

                for img_name in image_files:
                    img_path = os.path.join(page_path, img_name)

                    rel_path = os.path.join(
                        f"OdiaLineLevelDataSet/{book_id}_op/{page}",
                        img_name
                    )

                    tasks.append((img_path, rel_path, gt_lines))

            # 🔥 PARALLEL EXECUTION
            with Pool(NUM_WORKERS) as p:
                for result in p.imap_unordered(process_single, tasks):

                    if result is None:
                        continue

                    rel_path, gt_text = result

                    # write immediately
                    out_f.write(f"{rel_path}\t{gt_text}\n")
                    out_f.flush()


if __name__ == "__main__":
    main()