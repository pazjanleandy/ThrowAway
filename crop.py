import os
import cv2
import shutil
import argparse
import random

# Allowed image extensions for YOLO datasets
ALLOWED_EXTS = [".jpg", ".jpeg", ".png"]

def _load_class_map_from_yaml(base_dir: str):
    """Try to read names from a YOLO data YAML without requiring PyYAML.
    Searches for common yaml filenames and parses a minimal 'names:' list.
    Returns dict[id] = name or None if not found.
    """
    candidates = [
        os.path.join(base_dir, "data.yaml"),
        os.path.join(base_dir, "dataset.yaml"),
        os.path.join(base_dir, "data.yml"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            # Very light parsing: find 'names:' and list following in [ ... ]
            if "names:" not in text:
                continue
            # Normalize
            import re
            m = re.search(r"names\s*:\s*\[(.*?)\]", text, re.S)
            names = []
            if m:
                inside = m.group(1)
                parts = [p.strip().strip("'\"") for p in inside.split(',') if p.strip()]
                names = parts
            else:
                # Support block style
                block = text.split("names:", 1)[1]
                for line in block.splitlines()[1:]:
                    if not line.strip():
                        continue
                    if not line.startswith(" ") and not line.startswith("\t"):
                        break
                    line = line.strip()
                    if line.startswith("-"):
                        names.append(line.lstrip("- ").strip().strip("'\""))
                    else:
                        break
            if names:
                return {idx: name for idx, name in enumerate(names)}
        except Exception:
            pass
    return None

def save_crop_to_class_folder(image, class_id, base_dir='cropped', counters=None, class_map=None, subset: str | None = None):
    if class_map is None:
        class_map = {
            0: 'glass',
            1: 'metal',
            2: 'organic',
            3: 'paper',
            4: 'plastic',
        }
    # Unknown classes go to 'unsupported' instead of failing
    class_name = class_map.get(class_id, 'unsupported')
    # If subset specified ('train' or 'val'), write under that split
    class_folder = os.path.join(base_dir, subset, class_name) if subset else os.path.join(base_dir, class_name)
    os.makedirs(class_folder, exist_ok=True)

    # Determine proposed index: use per-run counter if provided, else find next available
    key = f"{subset}/{class_name}" if subset else class_name
    if counters is not None:
        proposed_idx = counters.get(key, 0) + 1
    else:
        # Fallback: scan for the next available numeric index
        proposed_idx = 1
        while True:
            probe = os.path.join(class_folder, f"{class_name}_{proposed_idx:05d}.jpg")
            if not os.path.exists(probe):
                break
            proposed_idx += 1

    # Build base filename and append '+' if collision occurs, repeating until unique
    base_name = f"{class_name}_{proposed_idx:05d}"
    filename = base_name + ".jpg"
    filepath = os.path.join(class_folder, filename)
    while os.path.exists(filepath):
        base_name += "+"
        filename = base_name + ".jpg"
        filepath = os.path.join(class_folder, filename)

    cv2.imwrite(filepath, image)
    if counters is not None:
        counters[key] = counters.get(key, 0) + 1
    return filepath

def _find_image_path(image_dir: str, stem: str):
    for ext in ALLOWED_EXTS:
        candidate = os.path.join(image_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None

def _resolve_split_dirs(yolo_base: str, subset: str):
    """Return (image_dir, label_dir) for the subset, trying common layouts."""
    candidates_images = [
        os.path.join(yolo_base, subset, 'images'),
        os.path.join(yolo_base, 'images', subset),
        os.path.join(yolo_base, 'image', subset),
    ]
    candidates_labels = [
        os.path.join(yolo_base, subset, 'labels'),
        os.path.join(yolo_base, 'labels', subset),
    ]
    image_dir = next((p for p in candidates_images if os.path.exists(p)), None)
    label_dir = next((p for p in candidates_labels if os.path.exists(p)), None)
    return image_dir, label_dir

def process_yolo_dataset(yolo_base='yolodataset', out_base='cropped', nosplit: bool = False):
    counters = {}
    class_map = _load_class_map_from_yaml(yolo_base)
    # Support both 'valid' and 'val' (exclude 'test' as requested)
    subsets = ['train', 'valid', 'val']
    seen_subset = set()
    for subset in subsets:
        # Skip duplicate val/valid if both exist and we've processed one
        if subset in ('valid', 'val') and ('valid' in seen_subset or 'val' in seen_subset):
            continue
        image_dir, label_dir = _resolve_split_dirs(yolo_base, subset)
        if not label_dir or not os.path.exists(label_dir):
            print(f"[i] Skipping subset '{subset}': labels dir not found")
            continue
        if not image_dir or not os.path.exists(image_dir):
            print(f"[i] Skipping subset '{subset}': images dir not found")
            continue
        seen_subset.add(subset)
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        if not label_files:
            print(f"[i] No .txt labels in {label_dir}")
            continue
        for label_file in os.listdir(label_dir):
            if not label_file.endswith('.txt'):
                continue
            stem = os.path.splitext(label_file)[0]
            image_path = _find_image_path(image_dir, stem)
            label_path = os.path.join(label_dir, label_file)
            if not os.path.exists(image_path):
                print(f"[!] Image not found for {label_file}")
                continue
            image = cv2.imread(image_path)
            if image is None:
                print(f"[!] Could not read image: {image_path}")
                continue
            img_h, img_w = image.shape[:2]
            with open(label_path, 'r') as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, x_center, y_center, width, height = map(float, parts)
                    x_center *= img_w
                    y_center *= img_h
                    width *= img_w
                    height *= img_h
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_w, x2), min(img_h, y2)
                    cropped = image[y1:y2, x1:x2]
                    if cropped.size == 0:
                        continue
                    out_subset = None if nosplit else ('val' if subset in ('val', 'valid') else 'train')
                    save_crop_to_class_folder(cropped, int(class_id), out_base, counters, class_map=class_map, subset=out_subset)
    print("\n✅ Done cropping and sorting all YOLO-annotated objects.")
    print("Summary of crops saved:")
    names = list(class_map.values()) if isinstance(class_map, dict) else ['glass','metal','organic','paper','plastic']
    if 'unsupported' not in names:
        names.append('unsupported')
    if nosplit:
        for k in names:
            print(f"  {k}: {counters.get(k, 0)}")
    else:
        for split in ['train', 'val']:
            for k in names:
                print(f"  {split}/{k}: {counters.get(f'{split}/{k}', 0)}")

def split_cropped_images(base_dir='cropped', split_ratio=0.8):
    categories = ['glass', 'metal', 'organic', 'paper', 'plastic']
    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        if not os.path.exists(cat_dir):
            continue
        images = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        for split, split_imgs in [('train', train_imgs), ('val', val_imgs)]:
            split_cat_dir = os.path.join(base_dir, split, category)
            os.makedirs(split_cat_dir, exist_ok=True)
            for img in split_imgs:
                src = os.path.join(cat_dir, img)
                dst = os.path.join(split_cat_dir, img)
                shutil.move(src, dst)
        # Remove the original category folder if empty
        if not os.listdir(cat_dir):
            os.rmdir(cat_dir)

def randomize_train_val(base_dir='cropped', split_ratio=0.8):
    """Reshuffle images across train/val to an 80/20 split per class.
    - If no split exists (flat folders), create the split first.
    - If split exists, pool per-class images and redistribute.
    """
    train_root = os.path.join(base_dir, 'train')
    val_root = os.path.join(base_dir, 'val')
    # If no split exists (flat folders), fall back to creating a split from flat
    if not os.path.isdir(train_root) or not os.path.isdir(val_root):
        print("[i] No existing train/val split detected; creating a randomized 80/20 split from flat folders.")
        split_cropped_images(base_dir=base_dir, split_ratio=split_ratio)
        print("\n✅ Randomized train/val split to 80/20 per class.")
        return
    # Discover categories from existing split directories
    categories = set()
    for root in (train_root, val_root):
        if os.path.isdir(root):
            for name in os.listdir(root):
                path = os.path.join(root, name)
                if os.path.isdir(path):
                    categories.add(name)
    categories = sorted(categories)
    pool_root = os.path.join(base_dir, '_randomize_pool')
    for category in categories:
        pool_dir = os.path.join(pool_root, category)
        os.makedirs(pool_dir, exist_ok=True)
        train_cat = os.path.join(train_root, category)
        val_cat = os.path.join(val_root, category)
        # Move all images from train and val into pool
        for src_dir in (train_cat, val_cat):
            if os.path.isdir(src_dir):
                for f in os.listdir(src_dir):
                    if not f.lower().endswith(tuple(ALLOWED_EXTS)):
                        continue
                    src = os.path.join(src_dir, f)
                    dst = os.path.join(pool_dir, f)
                    if os.path.exists(dst):
                        base, ext = os.path.splitext(f)
                        idx = 1
                        while True:
                            cand = os.path.join(pool_dir, f"{base}({idx}){ext}")
                            if not os.path.exists(cand):
                                dst = cand
                                break
                            idx += 1
                    shutil.move(src, dst)
        # Shuffle and split 80/20
        images = [f for f in os.listdir(pool_dir) if f.lower().endswith(tuple(ALLOWED_EXTS))]
        if not images:
            # Cleanup empty pool dir
            try:
                if not os.listdir(pool_dir):
                    os.rmdir(pool_dir)
            except Exception:
                pass
            continue
        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        os.makedirs(train_cat, exist_ok=True)
        os.makedirs(val_cat, exist_ok=True)
        for f in train_imgs:
            shutil.move(os.path.join(pool_dir, f), os.path.join(train_cat, f))
        for f in val_imgs:
            shutil.move(os.path.join(pool_dir, f), os.path.join(val_cat, f))
        # Remove pool/category if empty
        try:
            if not os.listdir(pool_dir):
                os.rmdir(pool_dir)
        except Exception:
            pass
    # Remove pool root if empty
    try:
        if os.path.isdir(pool_root) and not os.listdir(pool_root):
            os.rmdir(pool_root)
    except Exception:
        pass
    print("\n✅ Randomized train/val split to 80/20 per class.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop YOLO dataset; outputs mirror YOLO train/val layout.")
    parser.add_argument('--base', type=str, default='yolodataset', help='Path to YOLO dataset root (default: yolodataset)')
    parser.add_argument('--out', type=str, default='cropped', help='Output base directory for cropped images (default: cropped)')
    parser.add_argument('-split', action='store_true', help='(Deprecated) Random split; ignored since we mirror YOLO splits')
    parser.add_argument('-ratio', type=float, default=0.8, help='(Deprecated) Ratio used only if -split were active')
    parser.add_argument('-notsplit', action='store_true', help='Write crops under <out>/<category>/ with no train/val subfolders (alias for -nosplit)')
    parser.add_argument('-nosplit', action='store_true', help='Write crops under <out>/<category>/ with no train/val subfolders')
    parser.add_argument('-randomize', action='store_true', help='Reshuffle all images across <out>/train and <out>/val into an 80/20 split per class')
    args = parser.parse_args()
    # Both -notsplit and -nosplit do the same thing
    nosplit_flag = args.nosplit or args.notsplit
    # If user asked to randomize, do NOT crop again; just randomize and exit
    if args.randomize:
        randomize_train_val(base_dir=args.out, split_ratio=0.8)
        raise SystemExit(0)
    process_yolo_dataset(yolo_base=args.base, out_base=args.out, nosplit=nosplit_flag)
    if args.split:
        print("[i] -split specified but ignored: crops already written to train/val based on YOLO splits.")
    if nosplit_flag:
        print("[i] Crops written to flat category folders (no train/val split)")
