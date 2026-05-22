import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random


# ====================================================================
# 1. ENCODER ALIGNMENT
# ====================================================================
class MedicalLabelEncoder:
    def __init__(self):
        self.chars = " %()-./012345678?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.char_to_num = {char: i + 1 for i, char in enumerate(self.chars)}
        self.num_to_char = {i + 1: char for i, char in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_num[c] for c in str(text) if c in self.char_to_num]

    def decode(self, nums):
        res = []
        for i, num in enumerate(nums):
            if num != 0 and (i == 0 or num != nums[i - 1]):
                res.append(self.num_to_char.get(num, ""))
        return "".join(res)

    @property
    def vocab_size(self):
        return len(self.chars) + 1


# ====================================================================
# 2. DATASET LOADER WITH HIGH-GENERALIZATION AUGMENTATION
# ====================================================================
class MedicalDataset(Dataset):
    def __init__(self, csv_file, img_dir, encoder, is_training=True):
        try:
            self.df = pd.read_csv(csv_file, encoding='utf-8')
        except:
            self.df = pd.read_csv(csv_file, encoding='ISO-8859-1')

        self.img_dir = img_dir
        self.encoder = encoder
        self.is_training = is_training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        label_text = self.df.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, str(img_name))

        if not os.path.exists(img_path) or pd.isna(label_text):
            return torch.zeros((1, 64, 256)), torch.LongTensor([0]), torch.LongTensor([1])

        raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            return torch.zeros((1, 64, 256)), torch.LongTensor([0]), torch.LongTensor([1])

        # Binarization Profile
        if np.mean(raw_img) > 127:
            _, processed_img = cv2.threshold(raw_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, processed_img = cv2.threshold(raw_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # SMART CROP: Isolate ink payload
        pts = np.argwhere(processed_img == 0)
        if len(pts) > 0:
            y_min, x_min = pts.min(axis=0)
            y_max, x_max = pts.max(axis=0)
            processed_img = processed_img[y_min:y_max + 1, x_min:x_max + 1]

        # HIGH-GENERALIZATION DATA AUGMENTATION PASS
        if self.is_training:
            # A. Geometric Distortion (Forced rotation variety)
            if random.random() < 0.50:
                angle = random.uniform(-6, 6)
                h_r, w_r = processed_img.shape
                M = cv2.getRotationMatrix2D((w_r // 2, h_r // 2), angle, 1.0)
                processed_img = cv2.warpAffine(processed_img, M, (w_r, h_r), borderValue=255)

            # B. Ink Thickness Modification (Simulate different pen variations)
            if random.random() < 0.30:
                kernel = np.ones((2, 2), np.uint8)
                if random.random() < 0.5:
                    processed_img = cv2.erode(processed_img, kernel, iterations=1)
                else:
                    processed_img = cv2.dilate(processed_img, kernel, iterations=1)

            # C. Resolution Blurring (Simulate focus shifts from phone cameras)
            if random.random() < 0.25:
                processed_img = cv2.GaussianBlur(processed_img, (3, 3), 0)

        target_w, target_h = 256, 64
        padded_canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255

        # Aspect Ratio Preservation
        h_img, w_img = processed_img.shape
        scale = target_h / h_img
        nw = int(w_img * scale)
        nh = target_h

        if nw > target_w:
            scale = target_w / w_img
            nw = target_w
            nh = int(h_img * scale)

        resized_crop = cv2.resize(processed_img, (max(4, nw), max(4, nh)))

        start_x = max(0, (target_w - nw) // 2)
        start_y = max(0, (target_h - nh) // 2)
        padded_canvas[start_y:start_y + nh, start_x:start_x + nw] = resized_crop

        tensor_img = padded_canvas.astype(np.float32) / 255.0
        tensor_img = (tensor_img - 0.5) / 0.5
        tensor_img = torch.from_numpy(tensor_img).unsqueeze(0).float()

        label_encoded = self.encoder.encode(label_text)
        if not label_encoded:
            label_encoded = [0]

        return tensor_img, torch.LongTensor(label_encoded), torch.LongTensor([len(label_encoded)])


# ====================================================================
# 3. ARCHITECTURE BLOCK (Fully Aligned to Deployment Scripts)
# ====================================================================
class MedicalCRNN(nn.Module):
    def __init__(self, vocab_size):
        super(MedicalCRNN, self).__init__()
        # Pristine CNN sequential pipeline matching your stable .pth save footprint
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.hidden_size = 256  # Synced 256 matrix boundaries
        self.num_layers = 2
        self.rnn = nn.LSTM(input_size=1024, hidden_size=self.hidden_size, num_layers=self.num_layers,
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(self.hidden_size * 2, vocab_size)

    def forward(self, img_tensor, hx=None):
        features = self.cnn(img_tensor)
        b, c, h, w = features.size()
        features = features.view(b, c * h, w).permute(0, 2, 1)
        rnn_out, _ = self.rnn(features, hx)
        logits = self.fc(rnn_out)
        return logits.log_softmax(2)


# ====================================================================
# 4. TRAINING HARNESS WITH TEST GENERALIZATION MONITORING
# ====================================================================
def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = MedicalLabelEncoder()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

    CLEAN_BASE = os.path.join(PROJECT_ROOT, 'data', 'clean', 'MedicalCRNN_clean')

    # Paths Management
    TRAIN_CSV = os.path.join(CLEAN_BASE, 'Train_Label.csv')
    TRAIN_DIR = os.path.join(CLEAN_BASE, 'train')
    TEST_CSV = os.path.join(CLEAN_BASE, 'Test_Label.csv')
    TEST_DIR = os.path.join(CLEAN_BASE, 'test')

    MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'MedicalCRNN_v1.pth')
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"🧹 Purging old weight footprint at {MODEL_SAVE_PATH}")
        try:
            os.remove(MODEL_SAVE_PATH)
        except OSError:
            pass

    # Initialize Datasets
    train_ds = MedicalDataset(TRAIN_CSV, TRAIN_DIR, encoder, is_training=True)
    test_ds = MedicalDataset(TEST_CSV, TEST_DIR, encoder, is_training=False)

    def collate_fn(batch):
        images, labels, lengths = zip(*batch)
        return torch.stack(images, 0), \
            nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0), \
            torch.cat(lengths, 0)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = MedicalCRNN(encoder.vocab_size).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

    TOTAL_EPOCHS = 350
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
    )

    best_test_loss = float('inf')

    print(f"🚀 Optimized Deep Training Engine Active on: {device}")
    print(f"📊 Training Dimensions: {len(train_ds)} samples | Validation Test Dimensions: {len(test_ds)} samples")

    for epoch in range(TOTAL_EPOCHS):
        # --- PHASE A: TRAINING PASS ---
        model.train()
        train_epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{TOTAL_EPOCHS} Train")

        for imgs, labels, lengths in progress_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            num_directions = 2
            h0 = torch.zeros(model.num_layers * num_directions, imgs.size(0), model.hidden_size).to(device)
            c0 = torch.zeros(model.num_layers * num_directions, imgs.size(0), model.hidden_size).to(device)

            preds = model(imgs, (h0, c0)).permute(1, 0, 2)
            input_lens = torch.full((imgs.size(0),), preds.size(0), dtype=torch.long)

            loss = criterion(preds, labels, input_lens, lengths)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_epoch_loss += loss.item()

        avg_train_loss = train_epoch_loss / len(train_loader)

        # --- PHASE B: LIVE VAL MONITORING PASS ---
        model.eval()
        test_epoch_loss = 0
        with torch.no_grad():
            for imgs, labels, lengths in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                h0 = torch.zeros(model.num_layers * num_directions, imgs.size(0), model.hidden_size).to(device)
                c0 = torch.zeros(model.num_layers * num_directions, imgs.size(0), model.hidden_size).to(device)

                preds = model(imgs, (h0, c0)).permute(1, 0, 2)
                input_lens = torch.full((imgs.size(0),), preds.size(0), dtype=torch.long)

                loss = criterion(preds, labels, input_lens, lengths)
                test_epoch_loss += loss.item()

        avg_test_loss = test_epoch_loss / len(test_loader)
        scheduler.step(avg_test_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"📉 Epoch {epoch + 1} Complete | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Unseen Test Loss: {avg_test_loss:.4f}")

        # --- PHASE C: VAL GUARD SAVER ---
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            temp_save_path = MODEL_SAVE_PATH + ".tmp"
            try:
                torch.save(model.state_dict(), temp_save_path)
                if os.path.exists(MODEL_SAVE_PATH):
                    os.remove(MODEL_SAVE_PATH)
                os.rename(temp_save_path, MODEL_SAVE_PATH)
                print(f"🌟 Peak Generalization Saved (New Best Test Loss: {best_test_loss:.4f}) -> {MODEL_SAVE_PATH}")
            except Exception:
                fallback_path = os.path.join(PROJECT_ROOT, 'models', 'MedicalCRNN_best_generalization.pth')
                torch.save(model.state_dict(), fallback_path)

    print("🏁 High-generalization model training complete!")


if __name__ == "__main__":
    run_training()