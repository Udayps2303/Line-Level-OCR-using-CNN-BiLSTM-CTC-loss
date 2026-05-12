import re

INPUT_FILE = "output/map.txt"
OUTPUT_FILE = "output/map_sorted.txt"


def extract_keys(path):
    """
    Extract:
    book_id, page_number, line_number
    from path like:
    OdiaLineLevelDataSet/1_op/1_0001-012/1_0001-012_line_3.png
    """

    # Book ID
    book_match = re.search(r'/(\d+)_op/', path)
    book_id = int(book_match.group(1)) if book_match else 999

    # Page number (last part after '-')
    page_match = re.search(r'_(\d{4}-\d+)', path)
    if page_match:
        page_str = page_match.group(1)   # e.g. 0001-012
        page_num = int(page_str.split('-')[1])
    else:
        page_num = 9999

    # Line number
    line_match = re.search(r'line_(\d+)', path)
    line_num = int(line_match.group(1)) if line_match else 9999

    return book_id, page_num, line_num


def main():

    entries = []

    # Read file
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                img_path, gt_text = line.split('\t', 1)
                keys = extract_keys(img_path)

                entries.append((keys, img_path, gt_text))

            except:
                continue  # skip bad lines

    # Sort
    entries.sort(key=lambda x: (x[0][0], x[0][1], x[0][2]))

    # Write sorted output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for _, img_path, gt_text in entries:
            f.write(f"{img_path}\t{gt_text}\n")


if __name__ == "__main__":
    main()