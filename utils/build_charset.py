"""
utils/build_charset.py
Scans the full mapping file and writes data/charset.txt.
Run once before training:
    python utils/build_charset.py --config config.yaml
"""

import argparse
import os
import yaml
from tqdm import tqdm


def build_charset(mapping_file: str, out_path: str, max_label_len: int) -> list[str]:
    chars = set()
    skipped = 0
    total = 0

    print(f"Scanning mapping file: {mapping_file}")
    with open(mapping_file, encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            _, label = line.split("\t", 1)
            total += 1
            if len(label) > max_label_len:
                skipped += 1
                continue
            chars.update(label)

    charset = sorted(chars)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ch in charset:
            f.write(ch + "\n")

    print(f"Total lines   : {total}")
    print(f"Skipped (long): {skipped}")
    print(f"Unique chars  : {len(charset)}")
    print(f"Charset saved : {out_path}")
    return charset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    build_charset(
        mapping_file=cfg["dataset"]["mapping_file"],
        out_path=cfg["dataset"]["charset_file"],
        max_label_len=cfg["dataset"]["max_label_len"],
    )
