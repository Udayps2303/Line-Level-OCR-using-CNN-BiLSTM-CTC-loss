"""
FINAL Odia (Oriya) OCR Pipeline
Compatible with:
- Python 3.10
- macOS (M1/M2)
- VSCode
- System Tesseract (Homebrew)

FEATURES:
✔ Directory-based input (NO argparse)
✔ Automatic blank page detection & skipping
✔ 300 DPI PDF page extraction
✔ Image preprocessing (denoise + CLAHE + threshold)
✔ Line segmentation (OpenCV)
✔ Oriya OCR using Tesseract (ori)
✔ Ground truth validation (fuzzy matching)
✔ CSV per book: page_no, image_path, extracted_text, ground_truth, score
✔ Robust tessdata path fix for Mac (/opt/homebrew)

EXPECTED INPUT STRUCTURE:
input/
 └── books/
      ├── pdf/
      │     ├── book1.pdf
      │     ├── book2.pdf
      └── GT/
            ├── book1.txt
            ├── book2.txt
"""

"""
Memory-Optimized Odia OCR Pipeline
Safe for MacBook Air (8GB / 16GB RAM)
Python 3.10 compatible
"""

"""
Odia OCR Line Extraction Pipeline (Single Book Version)
Python 3.10 | Mac M1 | Memory Optimized | VSCode Friendly

Features:
- Processes ONE book (user provided path)
- 300 DPI page extraction (configurable)
- Blank page detection
- Line segmentation
- Odia OCR (Tesseract)
- Ground truth validation
- CSV generation
- Progress display (no more "stuck" feeling)
"""

import os
import gc
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
import pytesseract
from rapidfuzz import process, fuzz

# ==============================
# 🔧 USER INPUT (CHANGE THIS)
# ==============================
BOOK_PDF_PATH = "input/books/pdf/2.pdf"  # <-- GIVE YOUR BOOK PATH HERE
GT_TEXT_PATH = "input/books/GT/2.txt"    # <-- MATCHING GROUND TRUTH FILE

# ==============================
# TESSERACT CONFIG (Mac M1)
# ==============================
os.environ["TESSDATA_PREFIX"] = "/opt/homebrew/share/tessdata"

# ==============================
# CONFIGURATION
# ==============================
DPI = 250  # (Recommended: 200–250 for MacBook Air)
TESS_LANG = "ori"
FUZZY_THRESHOLD = 80
BLANK_WHITE_RATIO = 0.98

OUTPUT_ROOT = Path("output")
LINES_DIR = OUTPUT_ROOT / "lines"
CSV_DIR = OUTPUT_ROOT / "csv"


# ==============================
# UTILITY FUNCTIONS
# ==============================
def make_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def is_blank_page(gray_img):
    white_ratio = np.mean(gray_img > 240)
    variance = np.var(gray_img)
    return white_ratio > BLANK_WHITE_RATIO or variance < 5


def preprocess_page(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if is_blank_page(gray):
        return None, True

    # Denoising
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Contrast enhancement (very useful for old books)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Adaptive binarization
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25, 15
    )

    return binary, False


def segment_lines(binary_img):
    inv = 255 - binary_img
    h, w = inv.shape

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (w // 40 + 1, 3)
    )

    dilated = cv2.dilate(inv, kernel, iterations=2)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if 12 < bh < h // 2:  # Filter noise & big blocks
            boxes.append((x, y, bw, bh))

    return sorted(boxes, key=lambda b: b[1])


def ocr_line(pil_img):
    text = pytesseract.image_to_string(
        pil_img,
        lang=TESS_LANG,
        config="--psm 7"
    )
    return text.strip().replace("\n", " ")


# ==============================
# MAIN PIPELINE (SINGLE BOOK)
# ==============================
def run_single_book_pipeline():

    pdf_path = Path(BOOK_PDF_PATH)
    gt_path = Path(GT_TEXT_PATH)

    if not pdf_path.exists():
        print(f"ERROR: PDF not found -> {pdf_path}")
        return

    book_name = pdf_path.stem
    print(f"\n========== Processing Book: {book_name} ==========")

    # Create output folders
    book_line_dir = LINES_DIR / book_name
    make_dir(book_line_dir)
    make_dir(CSV_DIR)

    # Get total pages safely (NO memory load)
    try:
        info = pdfinfo_from_path(str(pdf_path))
        total_pages = info["Pages"]
    except Exception as e:
        print(f"Failed to read PDF info: {e}")
        return

    print(f"Total Pages: {total_pages}")

    records = []

    # ================= PAGE LOOP (MEMORY SAFE) =================
    for page_number in range(1, total_pages + 1):
        print(f"Processing Page {page_number}/{total_pages}...")

        try:
            pages = convert_from_path(
                str(pdf_path),
                dpi=DPI,
                first_page=page_number,
                last_page=page_number,
                thread_count=1
            )

            if not pages:
                print(f"Warning: Could not load page {page_number}")
                continue

            pil_page = pages[0]

        except Exception as e:
            print(f"Error on page {page_number}: {e}")
            continue

        # Preprocess page
        binary, is_blank = preprocess_page(pil_page)

        if is_blank:
            print(f"Skipped blank page {page_number}")
            del pil_page
            gc.collect()
            continue

        # Line segmentation
        boxes = segment_lines(binary)

        if len(boxes) == 0:
            print(f"No lines detected on page {page_number}")
            del pil_page
            gc.collect()
            continue

        # Process each line
        for idx, (x, y, w, h) in enumerate(boxes, start=1):
            crop = binary[y:y + h, x:x + w]
            crop_pil = Image.fromarray(255 - crop)

            line_filename = f"{page_number:04d}_line_{idx:03d}.png"
            line_path = book_line_dir / line_filename
            crop_pil.save(line_path)

            extracted_text = ocr_line(crop_pil)

            records.append({
                "page_no": page_number,
                "image_path": str(line_path),
                "extracted_text": extracted_text
            })

        # Free memory (VERY IMPORTANT for MacBook Air)
        del pil_page
        gc.collect()

    print("\nOCR Extraction Completed!")

    # Save extracted CSV
    extracted_df = pd.DataFrame(records)
    extracted_csv_path = CSV_DIR / f"{book_name}_extracted.csv"
    extracted_df.to_csv(extracted_csv_path, index=False, encoding="utf-8")
    print(f"Saved Extracted CSV: {extracted_csv_path}")

    # ================= VALIDATION WITH GROUND TRUTH =================
    if gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_lines = [line.strip() for line in f if line.strip()]
        print("Ground truth file loaded.")
    else:
        print("Ground truth file not found. Skipping validation.")
        gt_lines = []

    validated_rows = []

    for row in records:
        extracted = row["extracted_text"]

        if gt_lines and extracted:
            match = process.extractOne(
                extracted, gt_lines, scorer=fuzz.WRatio
            )
            best_gt = match[0]
            score = match[1]

            if score < FUZZY_THRESHOLD:
                best_gt = ""
        else:
            best_gt = ""
            score = 0

        validated_rows.append({
            "page_no": row["page_no"],
            "image_path": row["image_path"],
            "extracted_text": extracted,
            "ground_truth": best_gt,
            "score": score
        })

    validation_csv_path = CSV_DIR / f"{book_name}_validation.csv"
    pd.DataFrame(validated_rows).to_csv(
        validation_csv_path, index=False, encoding="utf-8"
    )

    print(f"Saved Validation CSV: {validation_csv_path}")
    print("\n========== PIPELINE FINISHED SUCCESSFULLY ==========")


if __name__ == "__main__":
    run_single_book_pipeline()