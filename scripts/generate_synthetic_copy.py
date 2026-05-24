import os
import random
import csv
import shutil
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ====================================================================
# 1. BULLETPROOF CONTINUOUS WORD RENDERER (PREVENTS "NOT.PNG" ARTIFACTS)
# ====================================================================
def draw_word_continuous(word_str, font_path, target_w=256, target_h=64):
    """
    Dynamically sizes and draws the entire word block to guarantee it never
    clips or shrinks down to microscopic fragments like seen in not.PNG.
    """
    # Create target canvas frame
    base_canvas = Image.new("L", (target_w, target_h), 255)
    draw = ImageDraw.Draw(base_canvas)

    # 🟢 FIXED: Dynamic Font Fitting Loop - Scales font to fit the width perfectly
    font_size = 34 if len(word_str) <= 12 else 26
    font = None

    while font_size > 12:
        try:
            font = ImageFont.truetype(font_path, font_size)
            text_w = draw.textlength(word_str, font=font)
            # Ensure text has a safe horizontal padding margin
            if text_w < (target_w - 24):
                break
        except Exception:
            font = ImageFont.load_default()
        font_size -= 2

    if font is None:
        font = ImageFont.load_default()

    # Calculate exact rendering offsets using true text length metrics
    try:
        text_w = draw.textlength(word_str, font=font)
        # Handle older Pillow bounding box targets if textlength is missing
        text_h = font.getbbox(word_str)[3] - font.getbbox(word_str)[1] if hasattr(font, "getbbox") else font_size
    except Exception:
        text_w = len(word_str) * (font_size * 0.55)
        text_h = font_size

    # Perfect geometric center positioning
    start_x = int(max(12, (target_w - text_w) // 2))
    start_y = int(max(4, (target_h - text_h) // 2 - 2))

    # Draw solid continuous string sequence
    draw.text((start_x, start_y), word_str, fill=0, font=font)

    # Apply a gentle wrist angle tilt directly to the canvas frame bounds
    word_rot = random.uniform(-2.5, 2.5)
    base_canvas = base_canvas.rotate(word_rot, resample=Image.BICUBIC, expand=False, fillcolor=255)

    return base_canvas


# ====================================================================
# 2. CONTINUOUS GEOMETRIC & FLUID TREMOR DISTORTION MATRIX
# ====================================================================
def apply_handwriting_distortions(img_np):
    canvas_h, canvas_w = img_np.shape

    # A. Controlled Wave Warp: Gentle wavy baseline alignment matching l.PNG
    if random.random() < 0.70:
        rows, cols = img_np.shape
        map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))

        # Tamed down amplitude to ensure letters never get torn or broken up
        amplitude = random.uniform(0.6, 1.3)
        wavelength = random.uniform(40.0, 60.0)
        map_y = map_y + amplitude * np.sin(map_x / wavelength)

        img_np = cv2.remap(img_np, map_x.astype(np.float32), map_y.astype(np.float32),
                           interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    # B. Uniform Cursive Slant (Shear)
    if random.random() < 0.65:
        shear_factor = random.uniform(-0.12, 0.12)
        M_shear = np.float32([[1, shear_factor, 0], [0, 1, 0]])
        img_np = cv2.warpAffine(img_np, M_shear, (canvas_w, canvas_h), borderValue=255)

    # C. Bold Ink Profile (Simulates natural pen pressure varieties)
    if random.random() < 0.40:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        img_np = cv2.erode(img_np, kernel, iterations=1)

    return img_np


def apply_ink_bleed(img_np):
    if random.random() < 0.45:
        blurred = cv2.GaussianBlur(img_np, (3, 3), 0)
        _, img_np = cv2.threshold(blurred, random.randint(210, 230), 255, cv2.THRESH_BINARY)
    return img_np


def apply_paper_texture_and_camera_vignette(img_np):
    h, w = img_np.shape

    # Crisp clean paper base color layer
    base_paper = np.ones((h, w), dtype=np.uint8) * random.randint(246, 252)
    grain_noise = np.random.randint(-2, 2, (h, w)).astype(np.int16)
    base_paper = cv2.add(base_paper, grain_noise, dtype=cv2.CV_8U)

    text_mask = img_np.astype(float) / 255.0
    processed_canvas = (base_paper.astype(float) * text_mask).astype(np.uint8)

    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X ** 2 + Y ** 2)

    # 🟢 FIXED: Expanded vignette variance (sigma) to keep backgrounds bright and clear
    sigma = random.uniform(1.8, 2.4)
    light_gradient = np.exp(-(radius ** 2) / (2 * sigma ** 2))
    light_gradient = (light_gradient - light_gradient.min()) / (light_gradient.max() - light_gradient.min())

    shadow_factor = 0.82 + 0.18 * light_gradient
    return (processed_canvas.astype(float) * shadow_factor).astype(np.uint8)


# ====================================================================
# 3. SPLIT PRODUCTION EXECUTION ENGINE WITH ORIGINAL INCLUSION
# ====================================================================
def generate_split_synthetic_dataset(target_total=40000, train_ratio=0.80):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Environment directory targets linked directly to pppp.PNG parameters
    RAW_DIR = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System\data\raw\RxHandBD-Raw"
    CLEAN_BASE = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System\data\clean\MedicalCRNN_clean"

    EXCEL_FILE_XLSX = os.path.join(RAW_DIR, "Prescription_Labels.xlsx")
    EXCEL_FILE_XLS = os.path.join(RAW_DIR, "Prescription_Labels.xls")
    ORIG_IMAGES_FOLDER = os.path.join(RAW_DIR, "Images")

    TRAIN_IMG_DIR = os.path.join(CLEAN_BASE, "train_synthetic")
    TEST_IMG_DIR = os.path.join(CLEAN_BASE, "test_synthetic")
    TRAIN_CSV = os.path.join(CLEAN_BASE, "Train_Label_Synthetic_Train.csv")
    TEST_CSV = os.path.join(CLEAN_BASE, "Train_Label_Synthetic_Test.csv")

    FONT_FOLDER = os.path.join(SCRIPT_DIR, "handwritten_fonts")

    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(TEST_IMG_DIR, exist_ok=True)
    os.makedirs(FONT_FOLDER, exist_ok=True)

    if os.path.exists(EXCEL_FILE_XLSX):
        print(f"📖 Ingesting Excel source catalog matrix:\n👉 {EXCEL_FILE_XLSX}")
        df_vocab = pd.read_excel(EXCEL_FILE_XLSX)
    elif os.path.exists(EXCEL_FILE_XLS):
        print(f"📖 Ingesting Excel source catalog matrix:\n👉 {EXCEL_FILE_XLS}")
        df_vocab = pd.read_excel(EXCEL_FILE_XLS)
    else:
        print(f"❌ Error: Cannot locate 'Prescription_Labels' excel sheet inside raw data directory.")
        return

    img_col = df_vocab.columns[0]
    text_col = df_vocab.columns[1]
    for col in df_vocab.columns:
        col_lower = str(col).lower()
        if "file" in col_lower or "image" in col_lower or "name" in col_lower:
            img_col = col
        if "text" in col_lower or "label" in col_lower or "word" in col_lower or "medicine" in col_lower:
            text_col = col

    unique_medicines = df_vocab[text_col].dropna().astype(str).unique().tolist()
    num_unique_words = len(unique_medicines)
    print(f"🎯 Loaded {num_unique_words} vocab paths from column header: [{text_col}]")

    target_train_total = int(target_total * train_ratio)
    target_test_total = target_total - target_train_total
    count_per_word = int(np.ceil(target_total / num_unique_words))
    train_vars_per_word = int(np.ceil(count_per_word * train_ratio))

    with open(TRAIN_CSV, mode='w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(["FILENAME", "TEXT"])
    with open(TEST_CSV, mode='w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(["FILENAME", "TEXT"])

    # 🟢 FIXED: Prioritize scalable TrueType system files to block un-scalable micro fallbacks
    valid_fonts = [os.path.join(FONT_FOLDER, f) for f in os.listdir(FONT_FOLDER) if f.lower().endswith('.ttf')]
    if not valid_fonts:
        FALLBACK_PATHS = ["C:\\Windows\\Fonts\\segoeui.ttf", "C:\\Windows\\Fonts\\arial.ttf"]
        valid_fonts = [f for f in FALLBACK_PATHS if os.path.exists(f)]

    print(f"⚡ Allocation Matrix Goals: Train={target_train_total} | Test={target_test_total}")
    global_train_counter = 0
    global_test_counter = 0

    # PHASE 1: Generate High-Legibility Continuous Synthetic Text Streams
    with open(TRAIN_CSV, mode='a', newline='', encoding='utf-8') as tr_f, \
            open(TEST_CSV, mode='a', newline='', encoding='utf-8') as ts_f:

        train_writer = csv.writer(tr_f)
        test_writer = csv.writer(ts_f)

        for word in tqdm(unique_medicines, desc="Generating Connected Script"):
            word_str = str(word).strip()
            if not word_str or word_str.lower() in ["text", "label", "none", "nan"]:
                continue

            for var_idx in range(count_per_word):
                is_train_split = var_idx < train_vars_per_word

                if is_train_split and global_train_counter >= target_train_total:
                    is_train_split = False
                if not is_train_split and global_test_counter >= target_test_total:
                    if global_train_counter < target_train_total:
                        is_train_split = True
                    else:
                        break

                # Pick a random font track path from your collection
                chosen_font_path = random.choice(valid_fonts) if valid_fonts else "arial.ttf"

                # Run dynamic metric drawing pass
                pil_canvas = draw_word_continuous(word_str, chosen_font_path, target_w=256, target_h=64)
                img_np = np.array(pil_canvas)

                img_np = apply_handwriting_distortions(img_np)
                img_np = apply_ink_bleed(img_np)
                img_np = apply_paper_texture_and_camera_vignette(img_np)

                if is_train_split:
                    filename = f"{global_train_counter:05d}.jpg"
                    save_path = os.path.join(TRAIN_IMG_DIR, filename)
                    cv2.imwrite(save_path, img_np)
                    train_writer.writerow([filename, word_str])
                    global_train_counter += 1
                else:
                    filename = f"{global_test_counter:05d}.jpg"
                    save_path = os.path.join(TEST_IMG_DIR, filename)
                    cv2.imwrite(save_path, img_np)
                    test_writer.writerow([filename, word_str])
                    global_test_counter += 1

            if (global_train_counter + global_test_counter) >= target_total:
                break

    # PHASE 2: Duplicate Original Baseline Assets (l.PNG Sync Pattern)
    if os.path.exists(ORIG_IMAGES_FOLDER):
        print(f"\n📦 Merging pristine original reference files into output lanes...")

        df_shuffled = df_vocab.sample(frac=1.0, random_state=42).reset_index(drop=True)
        split_pivot_idx = int(len(df_shuffled) * train_ratio)

        real_train_copied = 0
        real_test_copied = 0

        with open(TRAIN_CSV, mode='a', newline='', encoding='utf-8') as tr_f, \
                open(TEST_CSV, mode='a', newline='', encoding='utf-8') as ts_f:

            train_writer = csv.writer(tr_f)
            test_writer = csv.writer(ts_f)

            for idx, row in tqdm(df_shuffled.iterrows(), total=len(df_shuffled), desc="Unifying Original Assets"):
                raw_filename = str(row[img_col]).strip()
                text_value = str(row[text_col]).strip()

                if not raw_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_filename = f"{raw_filename}.jpg"
                else:
                    src_filename = raw_filename

                src_path = os.path.join(ORIG_IMAGES_FOLDER, src_filename)
                if not os.path.exists(src_path):
                    continue

                dst_filename = f"orig_{src_filename}"

                if idx < split_pivot_idx:
                    shutil.copy2(src_path, os.path.join(TRAIN_IMG_DIR, dst_filename))
                    train_writer.writerow([dst_filename, text_value])
                    real_train_copied += 1
                else:
                    shutil.copy2(src_path, os.path.join(TEST_IMG_DIR, dst_filename))
                    test_writer.writerow([dst_filename, text_value])
                    real_test_copied += 1

        print(f"✅ Unified real files into target split pools: Train={real_train_copied} | Test={real_test_copied}")

    print(f"\n" + "=" * 60)
    print(f"🏆 SUCCESS: FIXED PIPELINE EXECUTED EXCELLENTLY")
    print(f"📂 High-legibility continuous cursive data safely compiled to: {CLEAN_BASE}")
    print("=" * 60)


if __name__ == "__main__":
    generate_split_synthetic_dataset(target_total=40000, train_ratio=0.80)