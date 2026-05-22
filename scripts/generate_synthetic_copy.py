import os
import random
import csv
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import shutil


# ====================================================================
# 1. ADVANCED HANDWRITING DISTORTION MATRIX
# ====================================================================
def apply_handwriting_distortions(img_np):
    canvas_h, canvas_w = img_np.shape

    # A. RANDOM SHEARING (Slanting text)
    if random.random() < 0.70:
        shear_factor = random.uniform(-0.22, 0.22)
        M_shear = np.float32([[1, shear_factor, 0], [0, 1, 0]])
        img_np = cv2.warpAffine(img_np, M_shear, (canvas_w, canvas_h), borderValue=255)

    # B. ELASTIC WAVE DISTORTION (Uneven human handwriting lines)
    if random.random() < 0.65:
        rows, cols = img_np.shape
        map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
        amplitude = random.uniform(1.2, 2.8)
        wavelength = random.uniform(22.0, 38.0)
        map_y = map_y + amplitude * np.sin(map_x / wavelength)
        img_np = cv2.remap(img_np, map_x.astype(np.float32), map_y.astype(np.float32),
                           interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    # C. VARIABLE INK PRESSURE
    if random.random() < 0.50:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        if random.random() < 0.50:
            img_np = cv2.erode(img_np, kernel, iterations=1)
        else:
            img_np = cv2.dilate(img_np, kernel, iterations=1)

            # D. SPECKLE ARTIFACT NOISE
    if random.random() < 0.25:
        noise_mask = np.random.rand(*img_np.shape) < 0.008
        img_np[noise_mask] = 0

    return img_np


# ====================================================================
# 2. BULK EXECUTION ENGINE (EXACT MATCH FOR YOUR FIXED PATHS)
# ====================================================================
def generate_isolated_synthetic_data(count_per_word=10):
    # Find absolute folder location of the script: C:\Users\Bubu\AI-Healthcare-Diagnostic-System\scripts
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Move one folder up to root: C:\Users\Bubu\AI-Healthcare-Diagnostic-System
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

    # Target Target Directory: C:\Users\Bubu\AI-Healthcare-Diagnostic-System\data\clean\MedicalCRNN_clean
    CLEAN_BASE = os.path.join(PROJECT_ROOT, "data", "clean", "MedicalCRNN_clean")

    # --- FIXED PATH INPUT LOOKUPS ---
    ORIGINAL_CSV = os.path.join(CLEAN_BASE, "Train_Label.csv")  # Read from MedicalCRNN_clean
    ORIGINAL_IMG_DIR = os.path.join(CLEAN_BASE, "train")  # Read original train images

    # --- NEW COMBINED DESTINATIONS (Saved right inside MedicalCRNN_clean) ---
    NEW_CSV = os.path.join(CLEAN_BASE, "Train_Label_With_Synthetic.csv")
    NEW_IMG_DIR = os.path.join(CLEAN_BASE, "train_synthetic_combined")

    # Fonts folder stays inside your scripts folder for easy access
    FONT_FOLDER = os.path.join(SCRIPT_DIR, "handwritten_fonts")

    # Create required directory targets safely
    os.makedirs(NEW_IMG_DIR, exist_ok=True)
    os.makedirs(FONT_FOLDER, exist_ok=True)

    # Hard-stop verification checkpoint
    if not os.path.exists(ORIGINAL_CSV):
        print(f"❌ Error: Cannot find your source file at target path:\n👉 {ORIGINAL_CSV}")
        return

    # STEP 1: Copy your original untouched spreadsheet records directly into the new file copy
    print(f"📋 Initializing new combined spreadsheet list at:\n👉 {NEW_CSV}")
    shutil.copyfile(ORIGINAL_CSV, NEW_CSV)

    # STEP 2: Clone your existing real training images into the new combined playground folder
    if os.path.exists(ORIGINAL_IMG_DIR):
        print(f"📦 Cloning original real images into the new combined dataset directory...")
        for filename in tqdm(os.listdir(ORIGINAL_IMG_DIR), desc="Migrating Real Images"):
            src_file = os.path.join(ORIGINAL_IMG_DIR, filename)
            dst_file = os.path.join(NEW_IMG_DIR, filename)
            if os.path.isfile(src_file) and not os.path.exists(dst_file):
                shutil.copyfile(src_file, dst_file)
    else:
        print(
            f"⚠️ Note: Original image folder not found at {ORIGINAL_IMG_DIR}. Continuing with synthetic generations only.")

    # STEP 3: Read unique medicine names from the original dataset matrix
    df = pd.read_csv(ORIGINAL_CSV)
    unique_medicines = df['Text'].dropna().unique().tolist()
    print(f"🎯 Successfully extracted {len(unique_medicines)} unique names to replicate!")

    # STEP 4: Fonts Integrity Verification
    valid_fonts = [os.path.join(FONT_FOLDER, f) for f in os.listdir(FONT_FOLDER) if f.lower().endswith('.ttf')]
    if not valid_fonts:
        print(f"\n⚠️ Warning: No custom handwritten fonts found inside: {FONT_FOLDER}")
        print("👉 Falling back to default system typography fonts temporarily...")

        FALLBACK_PATHS = [
            "C:\\Windows\\Fonts\\arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]
        valid_fonts = [f for f in FALLBACK_PATHS if os.path.exists(f)]

    # STEP 5: Generate Handwritten Image Matrix Mutators
    print(f"⚡ Synthesis Engine active: Building {count_per_word} variations for each word entry...")
    file_counter = 0

    # Open the newly created CSV tracking file in append ('a') mode to seamlessly add rows to the bottom
    with open(NEW_CSV, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        for word in tqdm(unique_medicines, desc="Generating Script Variations"):
            word_str = str(word).strip()
            if not word_str or word_str.lower() == "text":  # skip potential duplicate headers
                continue

            for var_idx in range(count_per_word):
                canvas_w, canvas_h = 256, 64
                pil_img = Image.new("L", (canvas_w, canvas_h), 255)
                draw = ImageDraw.Draw(pil_img)

                if len(word_str) > 12:
                    font_size = random.randint(18, 22)
                else:
                    font_size = random.randint(24, 32)

                if valid_fonts:
                    try:
                        font = ImageFont.truetype(random.choice(valid_fonts), font_size)
                    except:
                        font = ImageFont.load_default()
                else:
                    font = ImageFont.load_default()

                try:
                    text_w = draw.textlength(word_str, font=font)
                except:
                    text_w = len(word_str) * (font_size * 0.55)
                text_h = font_size

                start_x = int(max(4, (canvas_w - text_w) // 2))
                start_y = int(max(4, (canvas_h - text_h) // 2))

                draw.text((start_x, start_y), word_str, fill=0, font=font)

                img_np = np.array(pil_img)
                img_np = apply_handwriting_distortions(img_np)

                filename = f"generated_handwritten_{file_counter}_{random.randint(1000, 9999)}.jpg"
                full_save_path = os.path.join(NEW_IMG_DIR, filename)
                cv2.imwrite(full_save_path, img_np)

                writer.writerow([filename, word_str])
                file_counter += 1

    print(f"\n" + "=" * 60)
    print(f"🏆 SUCCESS: SAFE ISOLATED DATASET MATRIX BUILT")
    print(
        f"👉 Everything saved inside: C:\\Users\\Bubu\\AI-Healthcare-Diagnostic-System\\data\\clean\\MedicalCRNN_clean\\")
    print(f"👉 Combined Image Folder: train_synthetic_combined")
    print(f"👉 Combined Spreadsheet Log: Train_Label_With_Synthetic.csv")
    print(f"📊 Total New Handwriting Variants Created: {file_counter}")
    print("=" * 60)


if __name__ == "__main__":
    generate_isolated_synthetic_data(count_per_word=10)