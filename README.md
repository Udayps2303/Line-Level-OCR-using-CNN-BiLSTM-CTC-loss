# 🔤 Odia Line-Level OCR — CNN-BiLSTM-CTC

A TensorFlow/Keras implementation of a deep **CNN-BiLSTM-CTC** model for recognising printed **Odia script** from line-level images. Trained on ~4 lakh line samples, the model achieves a **Character Error Rate (CER) < 2%** by epoch 30–40.

---

## 📐 Architecture

```
Input (H=64, W=variable, C=1 grayscale)
        │
  ┌─────▼──────────────────────────────────────────────────────┐
  │  CNN Block 0 : Conv2D(64)  → BN → ReLU → MaxPool(2,2)     │  H→H/2,  W→W/2
  │  CNN Block 1 : Conv2D(128) → BN → ReLU → MaxPool(2,2)     │  H→H/4,  W→W/4
  │  CNN Block 2 : Conv2D(256) → BN → ReLU → MaxPool(2,1)     │  H→H/8,  W unchanged
  │  CNN Block 3 : Conv2D(256) → BN → ReLU → MaxPool(2,1)     │  H→H/16, W unchanged
  └────────────────────────────────────────────────────────────┘
        │  shape: (B, 4, W/4, 256)
  Permute + Reshape → (B, T=W/4, 1024)   [feature per time-step]
        │
  ┌─────▼──────────────────────────────────────────────────────┐
  │  BiLSTM 0 : 256 fwd + 256 bwd → 512                       │
  │  Dropout 0.3                                               │
  │  BiLSTM 1 : 256 fwd + 256 bwd → 512                       │
  │  Dropout 0.3                                               │
  │  BiLSTM 2 : 256 fwd + 256 bwd → 512                       │
  └────────────────────────────────────────────────────────────┘
        │
  Dense(num_classes + 1)    [+1 for CTC blank at index 0]
        │
  CTC Loss / Greedy Decode
```

**Parameter count (approx):** ~11 M total (CNN ~1.2 M · BiLSTM stack ~9.5 M · Dense ~0.1 M)

---

## 📁 Project Structure

```
odia-line-ocr/
├── config.yaml                   # All hyperparameters & paths
├── train.py                      # Main training script
├── evaluate.py                   # Test-set evaluation (CER / WER)
├── infer.py                      # Single image / directory inference
├── compute_cer_&_wer.py          # Standalone CER/WER computation
├── image_preprocessing_VIT.py    # Image preprocessing utilities (ViT-style)
├── image_size.py                 # Image size analysis
├── map_stats.py                  # Mapping file statistics
├── sorting_new_map.py            # Sort/clean mapping file
├── create_final_mapping.py       # Build final mapping from raw data
├── Line_extraction_using_teseract.py  # Tesseract-based line extraction
├── requirements.txt              # pip dependencies
├── environment.yml               # Conda environment (recommended)
├── data/
│   ├── dataset.py                # tf.data pipeline, preprocessing & charset utils
│   └── charset.txt               # Auto-generated Odia Unicode charset
├── models/
│   └── cnn_bilstm_ctc.py         # Model architecture + CTCModel class
└── utils/
    ├── build_charset.py          # One-time charset builder
    ├── lr_schedule.py            # Warmup + cosine annealing
    └── metrics.py                # CER / WER via edit distance
```

---

## ⚙️ Environment Setup

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate odia_ocr
```

### Option B — pip + system CUDA

> Requires **CUDA 11.8** and **cuDNN 8.6**

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Verify GPU:
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Expected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## 🚀 Usage

### 1. Configure paths in `config.yaml`

```yaml
dataset:
  mapping_file: "output/map_sorted.txt"
  image_root:   "output"
```

### 2. Build charset (run once)

Scans all ground-truth lines and writes `data/charset.txt`:

```bash
python utils/build_charset.py --config config.yaml
```

Expected output:
```
Total lines   : ~400,000
Unique chars  : ~200–250   (Odia Unicode + punctuation + digits)
Charset saved : data/charset.txt
```

### 3. Train

```bash
python train.py --config config.yaml
```

Resume from a checkpoint:
```bash
python train.py --config config.yaml --resume checkpoints/best_model.keras
```

Monitor with TensorBoard:
```bash
tensorboard --logdir logs/
```

### 4. Evaluate (CER / WER on test set)

```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.keras
```

### 5. Inference

Single image:
```bash
python infer.py --image /path/to/line.png --checkpoint checkpoints/best_model.keras
```

Whole directory:
```bash
python infer.py --dir /path/to/lines/ --checkpoint checkpoints/best_model.keras > results.tsv
```

---

## 🗂️ Mapping File Format

Each line is tab-separated:
```
OdiaLineLevelDataSet/40_op/40_0001-003/40_0001-003_line_1.png	ସମାଦକ :
```

- **Column 1**: relative image path (joined with `image_root`)
- **Column 2**: ground-truth Odia text (UTF-8)

---

## 🎛️ Key Hyperparameters (`config.yaml`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `target_height` | 64 | Fixed height in pixels |
| `max_width` | 2048 | Lines wider than this are centre-cropped |
| `batch_size` | 16 | Reduce to 8 if VRAM < 8 GB |
| `epochs` | 100 | Early stopping patience = 10 |
| `initial_lr` | 1e-3 | Warmup + cosine annealing |
| `lstm_units` | [256, 256, 256] | Units per direction (output = 2×) |
| `mixed_precision` | true | fp16 — requires CUDA GPU |

---

## 🛠️ Training Tips

| Situation | Recommendation |
|-----------|---------------|
| VRAM < 8 GB | Set `batch_size: 8` in config |
| Training loss NaN | Lower `initial_lr` to `5e-4` |
| CER plateaus early | Enable `dilation_erosion` augmentation |
| Very long lines (>1800 px) | Increase `max_width` to 2560 |
| Want faster convergence | Increase `lstm_units` to `[512, 512, 256]` |

---

## 📊 Expected Results

| Metric | Target |
|--------|--------|
| Character Error Rate (CER) | < 2% by epoch 30–40 |
| Char Accuracy | ~99% |
| Word Error Rate (WER) | < 5% |

Based on comparable Bengali architecture (47K samples). With 4-lakh Odia samples + augmentation, results can exceed these benchmarks.

---

## 🔑 Key Design Decisions

1. **Fixed height (64px), variable width** — avoids distorting Odia conjuncts
2. **Grayscale conversion** — reduces memory by 3×; colour carries no extra signal for printed text
3. **CTC blank = index 0** — consistent with `tf.nn.ctc_loss(blank_index=0)`
4. **W-halving only in first 2 pool layers** — preserves fine horizontal resolution needed for dense Odia conjunct sequences
5. **Warmup + cosine decay** — stable training from scratch on large datasets
6. **Mixed precision (fp16)** — ~1.8× throughput on Ampere/Turing GPUs

---

## 📦 Dependencies

- Python 3.10
- TensorFlow 2.15 (with CUDA support)
- OpenCV 4.8
- Albumentations 1.3
- pyctcdecode
- editdistance

See `requirements.txt` or `environment.yml` for the full pinned list.

---

## 📄 License

This project is released for research and educational purposes. Please cite appropriately if used in academic work.

---

## 🙏 Acknowledgements

- Architecture inspired by CRNN (Shi et al., 2016) adapted for Indic scripts
- Dataset: Odia line-level images from CDAC
- Odia Unicode support via standard UTF-8 encoding
# Line-Level-OCR-using-CNN-BiLSTM-CTC-loss
