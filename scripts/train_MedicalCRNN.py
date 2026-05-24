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
# 1. UTILITY: LEVENSHTEIN DISTANCE (For Lexicon Auto-Correction)
# ====================================================================
def get_levenshtein_distance(s1, s2):
    """Calculates the edit distance between two strings."""
    if len(s1) < len(s2):
        return get_levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def correct_prediction(pred_str, lexicon):
    """Snaps the model's raw prediction to the closest valid medical word."""
    if not pred_str:
        return ""
    if pred_str in lexicon:
        return pred_str

    best_match = pred_str
    min_dist = float('inf')
    for word in lexicon:
        dist = get_levenshtein_distance(pred_str, word)
        if dist < min_dist:
            min_dist = dist
            best_match = word
    return best_match


# ====================================================================
# 2. DYNAMIC ENCODER ALIGNMENT
# ====================================================================
class MedicalLabelEncoder:
    def __init__(self, label_files):
        """Dynamically harvests unique characters from a list of files (CSV or Excel)."""
        unique_chars = set()
        lexicon_set = set()

        for file_path in label_files:
            if os.path.exists(file_path):
                # Check extension to load properly
                if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                    df = pd.read_excel(file_path)
                else:
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(file_path, encoding='ISO-8859-1')

                # Update characters
                df['Text'].astype(str).apply(lambda x: unique_chars.update(list(x)))
                # Update medical lexicon for auto-correction
                lexicon_set.update(df['Text'].astype(str).unique().tolist())

        self.lexicon = lexicon_set
        self.chars = sorted(list(unique_chars))
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
# 3. HIGH-GENERALIZATION DATASET LOADER (Grayscale Preserved)
# ====================================================================
class MedicalDataset(Dataset):
    def __init__(self, file_path, img_dir, encoder, is_training=True):
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            self.df = pd.read_excel(file_path)
        else:
            try:
                self.df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                self.df = pd.read_csv(file_path, encoding='ISO-8859-1')

        self.img_dir = img_dir
        self.encoder = encoder
        self.is_training = is_training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        label_text = str(self.df.iloc[idx, 1])
        img_path = os.path.join(self.img_dir, str(img_name))

        if not os.path.exists(img_path) or pd.isna(label_text) or label_text == "nan":
            return torch.zeros((1, 64, 256)), torch.LongTensor([0]), torch.LongTensor([1]), ""

        raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            return torch.zeros((1, 64, 256)), torch.LongTensor([0]), torch.LongTensor([1]), ""

        # Smart Crop: Use thresholding ONLY to find ink coordinates, preserve original grayscale
        if np.mean(raw_img) > 127:
            _, thresh = cv2.threshold(raw_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(raw_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        pts = np.argwhere(thresh == 255)

        if len(pts) > 0:
            y_min, x_min = pts.min(axis=0)
            y_max, x_max = pts.max(axis=0)
            processed_img = raw_img[y_min:y_max + 1, x_min:x_max + 1]
        else:
            processed_img = raw_img

        # Distortions to prevent overfitting on small datasets
        if self.is_training:
            if random.random() < 0.40:
                angle = random.uniform(-5, 5)
                h_r, w_r = processed_img.shape
                M = cv2.getRotationMatrix2D((w_r // 2, h_r // 2), angle, 1.0)
                processed_img = cv2.warpAffine(processed_img, M, (w_r, h_r), borderValue=255)
            if random.random() < 0.20:
                processed_img = cv2.GaussianBlur(processed_img, (3, 3), 0)

        # Pad to 256x64 while preserving aspect ratio
        target_w, target_h = 256, 64
        padded_canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255

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

        # Normalize Grayscale [-1, 1]
        tensor_img = padded_canvas.astype(np.float32) / 255.0
        tensor_img = (tensor_img - 0.5) / 0.5
        tensor_img = torch.from_numpy(tensor_img).unsqueeze(0).float()

        label_encoded = self.encoder.encode(label_text)
        if not label_encoded:
            label_encoded = [0]

        return tensor_img, torch.LongTensor(label_encoded), torch.LongTensor([len(label_encoded)]), label_text


# ====================================================================
# 4. CUSTOM RESIDUAL BACKBONE (Built from scratch)
# ====================================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class MedicalResidualCRNN(nn.Module):
    def __init__(self, vocab_size):
        super(MedicalResidualCRNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer2 = ResidualBlock(64, 64, stride=1)
        self.pool1 = nn.MaxPool2d(2)

        self.layer3 = ResidualBlock(64, 128, stride=1)
        self.layer4 = ResidualBlock(128, 128, stride=1)
        self.pool2 = nn.MaxPool2d(2)

        self.layer5 = ResidualBlock(128, 256, stride=1)
        self.layer6 = ResidualBlock(256, 256, stride=1)
        self.pool3 = nn.MaxPool2d((2, 1))

        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d((2, 1))  # Final shape: [Batch, 512, 4, 64]

        self.hidden_size = 256
        self.rnn = nn.LSTM(input_size=512 * 4, hidden_size=self.hidden_size, num_layers=2,
                           bidirectional=True, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(self.hidden_size * 2, vocab_size)

    def forward(self, img_tensor, hx=None):
        x = self.layer1(img_tensor)
        x = self.layer2(x)
        x = self.pool1(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool2(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.pool3(x)
        x = self.layer7(x)
        x = self.pool4(x)

        b, c, h, w = x.size()
        x = x.view(b, c * h, w).permute(0, 2, 1)  # Yields -> [Batch, Sequence=64, Features=2048]
        rnn_out, _ = self.rnn(x, hx)
        logits = self.fc(rnn_out)
        return logits.log_softmax(2)


# ====================================================================
# 5. TRAINING HARNESS & LEX आईपीएस (EVALUATION)
# ====================================================================
def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

    CLEAN_BASE = os.path.join(PROJECT_ROOT, 'data', 'clean', 'MedicalCRNN_clean')

    # Restored to .csv extensions to match Windows hidden files
    TRAIN_CSV = os.path.join(CLEAN_BASE, 'Train_Label.csv')
    TEST_CSV = os.path.join(CLEAN_BASE, 'Test_Label.csv')

    TRAIN_DIR = os.path.join(CLEAN_BASE, 'train')
    TEST_DIR = os.path.join(CLEAN_BASE, 'test')

    MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'MedicalCRNN_v2_Residual.pth')
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # Initialize encoder using the CSVs
    encoder = MedicalLabelEncoder([TRAIN_CSV, TEST_CSV])

    train_ds = MedicalDataset(TRAIN_CSV, TRAIN_DIR, encoder, is_training=True)
    test_ds = MedicalDataset(TEST_CSV, TEST_DIR, encoder, is_training=False)

    def collate_fn(batch):
        images, labels, lengths, raw_texts = zip(*batch)
        return torch.stack(images, 0), \
            nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0), \
            torch.cat(lengths, 0), raw_texts

    # Added num_workers=0 to prevent multiprocessing lockups on Windows if they occur
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = MedicalResidualCRNN(encoder.vocab_size).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, min_lr=1e-6)

    TOTAL_EPOCHS = 350
    best_test_loss = float('inf')
    best_accuracy = 0.0

    print(f"🚀 From-Scratch Residual Engine Active on: {device}")
    print(f"📊 Lexicon Size: {len(encoder.lexicon)} unique words | Vocab Size: {encoder.vocab_size} characters")

    for epoch in range(TOTAL_EPOCHS):
        # --- PHASE A: TRAINING PASS ---
        model.train()
        train_epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{TOTAL_EPOCHS} Train")

        for imgs, labels, lengths, _ in progress_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            preds = model(imgs).permute(1, 0, 2)
            input_lens = torch.full((imgs.size(0),), preds.size(0), dtype=torch.long)

            loss = criterion(preds, labels, input_lens, lengths)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_epoch_loss += loss.item()

        avg_train_loss = train_epoch_loss / len(train_loader)

        # --- PHASE B: LIVE VAL MONITORING & LEXICON ACCURACY PASS ---
        model.eval()
        test_epoch_loss = 0
        correct_words = 0
        total_words = 0

        with torch.no_grad():
            for imgs, labels, lengths, raw_texts in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                preds = model(imgs).permute(1, 0, 2)
                input_lens = torch.full((imgs.size(0),), preds.size(0), dtype=torch.long)

                loss = criterion(preds, labels, input_lens, lengths)
                test_epoch_loss += loss.item()

                # Greedy Decode & Lexicon Correction
                _, max_indices = preds.max(2)  # [Seq_len, Batch]
                max_indices = max_indices.transpose(1, 0).cpu().numpy()  # [Batch, Seq_len]

                for i, seq in enumerate(max_indices):
                    raw_pred_str = encoder.decode(seq)

                    # Snap prediction to closest exact medical dictionary word
                    corrected_str = correct_prediction(raw_pred_str, encoder.lexicon)

                    ground_truth = raw_texts[i]
                    if corrected_str == ground_truth:
                        correct_words += 1
                    total_words += 1

        avg_test_loss = test_epoch_loss / len(test_loader)
        val_accuracy = (correct_words / total_words) * 100 if total_words > 0 else 0.0

        scheduler.step(avg_test_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"📉 Epoch {epoch + 1} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Accuracy: {val_accuracy:.2f}%")

        # --- PHASE C: VAL GUARD SAVER ---
        if avg_test_loss < best_test_loss or val_accuracy > best_accuracy:
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

            temp_save_path = MODEL_SAVE_PATH + ".tmp"
            torch.save(model.state_dict(), temp_save_path)
            if os.path.exists(MODEL_SAVE_PATH):
                os.remove(MODEL_SAVE_PATH)
            os.rename(temp_save_path, MODEL_SAVE_PATH)
            print(f"🌟 Peak Generalization Saved (Best Accuracy: {best_accuracy:.2f}%) -> {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    run_training()