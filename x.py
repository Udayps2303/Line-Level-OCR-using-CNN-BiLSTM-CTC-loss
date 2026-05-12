import os
import cv2
import torch
import random
import argparse
import warnings
import numpy as np

from tqdm import tqdm
from hi_sam.modeling.build import model_registry
from hi_sam.modeling.auto_mask_generator import AutoMaskGenerator

warnings.filterwarnings("ignore")


# ==========================================================
# Argument Parser
# ==========================================================
def get_args_parser():
    parser = argparse.ArgumentParser("Hi-SAM line extraction to white polygon crops")

    parser.add_argument(
        "--input",
        type=str,
        default=r"/home/cdac4070/Shomesh/Shubhanshu/Parthiv/output_pages",
        help="Path to input image or directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"/home/cdac4070/Shomesh/Shubhanshu/Parthiv/Hi_sam_final_lines",
        help="Directory to save final line images"
    )

    parser.add_argument("--model-type", dest="model_type", type=str, default="vit_h")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=r"/home/cdac4070/Shomesh/Shubhanshu/Parthiv/Hi-SAM/pretrained_checkpoint/line_detection_ctw1500.pth"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="ctw1500")
    parser.add_argument("--zero_shot", action="store_true")

    # Required by Hi-SAM
    parser.add_argument("--hier_det", default=True)
    parser.add_argument("--attn_layers", type=int, default=1)
    parser.add_argument("--prompt_len", type=int, default=12)
    parser.add_argument("--layout_thresh", type=float, default=0.5)
    parser.add_argument("--input-size", dest="input_size", nargs=2, type=int, default=[1024, 1024])

    parser.add_argument("--seed", type=int, default=42)

    # Output tuning
    parser.add_argument(
        "--padding",
        type=int,
        default=8,
        help="Extra white margin around each final cropped line"
    )
    parser.add_argument(
        "--mask-dilate",
        dest="mask_dilate",
        type=int,
        default=0,
        help="Dilate the detected line mask before polygon extraction. Keep 0 unless mask is too tight."
    )
    parser.add_argument(
        "--polygon-epsilon",
        dest="polygon_epsilon",
        type=float,
        default=0.002,
        help="Polygon simplification ratio. Smaller = more detailed contour."
    )
    parser.add_argument(
        "--save-combined",
        dest="save_combined",
        action="store_true",
        help="Also save one image containing all detected lines on white canvas"
    )
    parser.add_argument(
        "--save-debug",
        dest="save_debug",
        action="store_true",
        help="Also save contour overlay for debugging"
    )

    return parser.parse_args()


# ==========================================================
# Utilities
# ==========================================================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def normalize_mask(mask):
    """
    Hi-SAM masks often come as (1, H, W). Convert to (H, W) uint8 {0,1}.
    """
    m = np.asarray(mask)

    if m.ndim == 3:
        m = m[0]

    m = (m > 0).astype(np.uint8)
    return m


def load_image_paths(input_path):
    input_path = os.path.abspath(os.path.expanduser(input_path.strip()))
    print(f"[INFO] Input Path: {input_path}")

    if not os.path.exists(input_path):
        raise ValueError(f"[ERROR] Input path does not exist: {input_path}")

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    if os.path.isfile(input_path):
        if not input_path.lower().endswith(valid_ext):
            raise ValueError(f"[ERROR] Not a valid image file: {input_path}")
        image_list = [input_path]

    elif os.path.isdir(input_path):
        image_list = sorted([
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(valid_ext)
        ])

        if len(image_list) == 0:
            raise ValueError(f"[ERROR] No valid images found in directory: {input_path}")

    else:
        raise ValueError(f"[ERROR] Invalid input path: {input_path}")

    print(f"[INFO] Found {len(image_list)} image(s) to process.")
    return image_list


def sort_masks_top_to_bottom(masks):
    """
    Sort masks by median y-position so output line images are saved top-to-bottom.
    """
    mask_positions = []

    for mask in masks:
        m = normalize_mask(mask)
        ys, xs = np.where(m == 1)

        if len(ys) == 0:
            continue

        median_y = np.median(ys)
        mask_positions.append((m, median_y))

    mask_positions.sort(key=lambda x: x[1])
    return [item[0] for item in mask_positions]


def resolve_overlaps(sorted_masks):
    """
    Reassign overlapping pixels to the closest line center vertically.
    No pixel deletion.
    """
    if len(sorted_masks) == 0:
        return []

    mask_stack = np.stack(sorted_masks, axis=0)

    centers = []
    for m in sorted_masks:
        ys = np.where(m == 1)[0]
        centers.append(np.median(ys))

    centers = np.array(centers)

    pixel_count = np.sum(mask_stack, axis=0)
    overlap_positions = np.where(pixel_count > 1)

    for y, x in zip(*overlap_positions):
        claiming = np.where(mask_stack[:, y, x] == 1)[0]

        if len(claiming) <= 1:
            continue

        distances = np.abs(centers[claiming] - y)
        best = claiming[np.argmin(distances)]

        mask_stack[claiming, y, x] = 0
        mask_stack[best, y, x] = 1

    return [mask_stack[i] for i in range(mask_stack.shape[0])]


def mask_to_polygon(mask, mask_dilate=0, epsilon_ratio=0.002):
    """
    Convert binary mask to one external polygon.
    """
    m = (mask > 0).astype(np.uint8) * 255

    if mask_dilate > 0:
        k = 2 * mask_dilate + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.dilate(m, kernel, iterations=1)

    contours, _ = cv2.findContours(
        m,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    if epsilon_ratio > 0:
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)

    polygon = contour.reshape(-1, 2)

    if len(polygon) < 3:
        return None

    return polygon


def polygon_to_white_crop(image, polygon, padding=8):
    """
    Use polygon coordinates to copy only the segmented region onto a white canvas.
    No JSON is saved. Polygon is only used internally.
    """
    h, w = image.shape[:2]

    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [polygon.astype(np.int32)], 255)

    ys, xs = np.where(poly_mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        return None, None, None

    x1 = max(0, xs.min() - padding)
    y1 = max(0, ys.min() - padding)
    x2 = min(w - 1, xs.max() + padding)
    y2 = min(h - 1, ys.max() + padding)

    crop_img = image[y1:y2 + 1, x1:x2 + 1]
    crop_mask = poly_mask[y1:y2 + 1, x1:x2 + 1]

    white_canvas = np.full_like(crop_img, 255, dtype=np.uint8)
    white_canvas[crop_mask > 0] = crop_img[crop_mask > 0]

    return white_canvas, poly_mask, (x1, y1, x2, y2)


def save_line_images(cleaned_masks, image, save_root_dir, img_id, args):
    """
    Save only final PNG line images like the sample:
    - irregular polygon crop on white canvas
    - optional combined image
    - optional debug overlay
    """
    image_output_dir = os.path.join(save_root_dir, img_id)
    os.makedirs(image_output_dir, exist_ok=True)

    if len(cleaned_masks) == 0:
        print(f"[INFO] No usable masks for {img_id}")
        return

    combined = np.full_like(image, 255, dtype=np.uint8)
    debug_overlay = image.copy()

    saved_count = 0

    for i, m in enumerate(cleaned_masks):
        if np.count_nonzero(m) == 0:
            continue

        polygon = mask_to_polygon(
            mask=m,
            mask_dilate=args.mask_dilate,
            epsilon_ratio=args.polygon_epsilon
        )

        if polygon is None:
            print(f"[WARNING] Skipping line {i} in {img_id}: invalid polygon")
            continue

        final_crop, full_poly_mask, bbox = polygon_to_white_crop(
            image=image,
            polygon=polygon,
            padding=args.padding
        )

        if final_crop is None:
            print(f"[WARNING] Skipping line {i} in {img_id}: empty crop")
            continue

        save_path = os.path.join(image_output_dir, f"{img_id}_line_{saved_count:03d}.png")
        cv2.imwrite(save_path, cv2.cvtColor(final_crop, cv2.COLOR_RGB2BGR))
        saved_count += 1

        if args.save_combined:
            combined[full_poly_mask > 0] = image[full_poly_mask > 0]

        if args.save_debug:
            cv2.polylines(
                debug_overlay,
                [polygon.astype(np.int32)],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2
            )

    if args.save_combined:
        combined_path = os.path.join(image_output_dir, f"{img_id}_all_lines.png")
        cv2.imwrite(combined_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    if args.save_debug:
        debug_path = os.path.join(image_output_dir, f"{img_id}_debug_overlay.png")
        cv2.imwrite(debug_path, cv2.cvtColor(debug_overlay, cv2.COLOR_RGB2BGR))

    print(f"[INFO] Saved {saved_count} final line image(s) for {img_id}")


def get_dataset_params(dataset, zero_shot=False):
    if dataset == "totaltext":
        if zero_shot:
            fg_points_num = 50
            score_thresh = 0.3
        else:
            fg_points_num = 500
            score_thresh = 0.95

    elif dataset == "ctw1500":
        if zero_shot:
            fg_points_num = 100
            score_thresh = 0.6
        else:
            fg_points_num = 300
            score_thresh = 0.7
    else:
        raise ValueError("Unsupported dataset")

    return fg_points_num, score_thresh


# ==========================================================
# Main
# ==========================================================
def main():
    args = get_args_parser()
    set_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING] CUDA not available. Falling back to CPU.")
        args.device = "cpu"

    os.makedirs(args.output, exist_ok=True)

    # Load model
    hisam = model_registry[args.model_type](args)
    hisam.eval()
    hisam.to(args.device)
    print("[INFO] Loaded Hi-SAM model")

    amg = AutoMaskGenerator(hisam)

    fg_points_num, score_thresh = get_dataset_params(
        args.dataset,
        args.zero_shot
    )

    image_list = load_image_paths(args.input)

    for path in tqdm(image_list, desc="Processing"):
        img_id = os.path.splitext(os.path.basename(path))[0]

        image_bgr = cv2.imread(path)
        if image_bgr is None:
            print(f"[WARNING] Failed to read image: {path}")
            continue

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        try:
            amg.set_image(image)

            masks, scores = amg.predict_text_detection(
                from_low_res=False,
                fg_points_num=fg_points_num,
                batch_points_num=min(fg_points_num, 100),
                score_thresh=score_thresh,
                nms_thresh=0.3,
                zero_shot=args.zero_shot,
                dataset=args.dataset
            )

            if masks is None or len(masks) == 0:
                print(f"[INFO] No prediction for {img_id}")
                continue

            print(f"[INFO] Inference done for {img_id}. Raw masks: {len(masks)}")

            sorted_masks = sort_masks_top_to_bottom(masks)
            print(f"[INFO] Sorted masks: {len(sorted_masks)}")

            cleaned_masks = resolve_overlaps(sorted_masks)
            print(f"[INFO] Overlap resolved: {len(cleaned_masks)} masks")

            save_line_images(
                cleaned_masks=cleaned_masks,
                image=image,
                save_root_dir=args.output,
                img_id=img_id,
                args=args
            )

        except Exception as e:
            print(f"[ERROR] Failed on {img_id}: {e}")

    print("[INFO] Processing complete.")


if __name__ == "__main__":
    main()