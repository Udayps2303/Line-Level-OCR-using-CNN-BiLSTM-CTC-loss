def compute_avg_metrics(file_path):
    total_cer = 0.0
    total_wer = 0.0
    count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')

            # Ensure line has CER and WER
            if len(parts) < 4:
                continue

            try:
                cer = float(parts[-2].replace("CER:", ""))
                wer = float(parts[-1].replace("WER:", ""))

                # Skip invalid values
                if cer < 0 or wer < 0:
                    continue

                total_cer += cer
                total_wer += wer
                count += 1

            except ValueError:
                continue

    if count == 0:
        print("No valid entries found.")
        return

    avg_cer = total_cer / count
    avg_wer = total_wer / count

    print(f"Total valid samples: {count}")
    print(f"Average CER: {avg_cer:.4f}")
    print(f"Average WER: {avg_wer:.4f}")


# Example usage
file_path = "pred5.txt"  # replace with your file
compute_avg_metrics(file_path)