import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import pandas as pd
import os


# ====================================================================
# 1. UTILITY: LEVENSHTEIN DISTANCE (For Lexicon Auto-Correction)
# ====================================================================
def get_levenshtein_distance(s1, s2):
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
        unique_chars = set()
        lexicon_set = set()

        for file_path in label_files:
            if os.path.exists(file_path):
                if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                    df = pd.read_excel(file_path)
                else:
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(file_path, encoding='ISO-8859-1')

                df['Text'].astype(str).apply(lambda x: unique_chars.update(list(x)))
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
# 3. DATASET LOADER (Grayscale Preserved for Inference)
# ====================================================================
class MedicalDataset(Dataset):
    def __init__(self, file_path, img_dir, encoder):
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            self.df = pd.read_excel(file_path)
        else:
            try:
                self.df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                self.df = pd.read_csv(file_path, encoding='ISO-8859-1')
        self.img_dir = img_dir
        self.encoder = encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        label_text = str(self.df.iloc[idx, 1])
        img_path = os.path.join(self.img_dir, str(img_name))

        if not os.path.exists(img_path) or pd.isna(label_text) or label_text == "nan":
            return torch.zeros((1, 64, 256)), torch.LongTensor([0]), label_text

        raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            return torch.zeros((1, 64, 256)), torch.LongTensor([0]), label_text

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

        target_w, target_h = 256, 64
        padded_canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255

        h_img, w_img = processed_img.shape
        scale = target_h / float(h_img)
        nw = int(w_img * scale)
        nh = target_h

        if nw > target_w:
            scale = target_w / float(w_img)
            nw = target_w
            nh = int(h_img * scale)

        nw = max(4, nw)
        nh = max(4, nh)

        resized_crop = cv2.resize(processed_img, (nw, nh))
        start_x = max(0, (target_w - nw) // 2)
        start_y = max(0, (target_h - nh) // 2)
        padded_canvas[start_y:start_y + nh, start_x:start_x + nw] = resized_crop

        tensor_img = padded_canvas.astype(np.float32) / 255.0
        tensor_img = (tensor_img - 0.5) / 0.5
        tensor_img = torch.from_numpy(tensor_img).unsqueeze(0).float()

        label_encoded = self.encoder.encode(label_text)
        if not label_encoded:
            label_encoded = [0]

        return tensor_img, torch.LongTensor(label_encoded), label_text


# ====================================================================
# 4. CORRECTED ARCHITECTURE BLOCK (Residual Mirror)
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

        self.spatial_drop1 = nn.Dropout2d(p=0.2)

        self.layer5 = ResidualBlock(128, 256, stride=1)
        self.layer6 = ResidualBlock(256, 256, stride=1)
        self.pool3 = nn.MaxPool2d((2, 1))

        self.spatial_drop2 = nn.Dropout2d(p=0.2)

        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d((2, 1))

        self.hidden_size = 256
        self.num_layers = 2
        self.rnn = nn.LSTM(input_size=512 * 4, hidden_size=self.hidden_size, num_layers=self.num_layers,
                           bidirectional=True, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(self.hidden_size * 2, vocab_size)

    def forward(self, img_tensor, hx=None):
        x = self.layer1(img_tensor)
        x = self.layer2(x)
        x = self.pool1(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool2(x)
        x = self.spatial_drop1(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.pool3(x)
        x = self.spatial_drop2(x)
        x = self.layer7(x)
        x = self.pool4(x)

        b, c, h, w = x.size()
        x = x.view(b, c * h, w).permute(0, 2, 1)
        rnn_out, _ = self.rnn(x, hx)
        logits = self.fc(rnn_out)
        return logits.log_softmax(2)


# ====================================================================
# 5. BATCHED PERFORMANCE VERIFICATION PIPELINE
# ====================================================================
def verify_on_test_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
    CLEAN_BASE = os.path.join(PROJECT_ROOT, 'data', 'clean', 'MedicalCRNN_clean')

    TRAIN_CSV = os.path.join(CLEAN_BASE, 'Train_Label.csv')
    TEST_CSV = os.path.join(CLEAN_BASE, 'Test_Label.csv')
    TEST_DIR = os.path.join(CLEAN_BASE, 'test')

    MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'MedicalCRNN_v2_Residual.pth')

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"❌ ERROR: Trained weight file not found at: {MODEL_SAVE_PATH}")
        return

    encoder = MedicalLabelEncoder([TRAIN_CSV, TEST_CSV])
    eval_ds = MedicalDataset(TEST_CSV, TEST_DIR, encoder)

    def collate_fn(batch):
        images, labels, raw_texts = zip(*batch)
        return torch.stack(images, 0), nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0), raw_texts

    eval_loader = DataLoader(eval_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    print(f"🔄 Loading Upgraded Residual Architecture from: {MODEL_SAVE_PATH}")
    model = MedicalResidualCRNN(encoder.vocab_size).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    correct_predictions = 0
    total_samples = 0
    sample_print_counter = 0

    print(f"\n🔍 Running Model Recognition on {len(eval_ds)} UNSEEN Test Images:\n" + "=" * 50)

    with torch.no_grad():
        for imgs, labels, raw_texts in eval_loader:
            imgs = imgs.to(device)

            preds = model(imgs).permute(1, 0, 2)
            _, max_indices = preds.max(2)
            max_indices = max_indices.transpose(1, 0).cpu().numpy()

            for b_idx in range(imgs.size(0)):
                if torch.sum(imgs[b_idx]) == 0:
                    continue

                raw_pred_str = encoder.decode(max_indices[b_idx]).strip()
                corrected_str = correct_prediction(raw_pred_str, encoder.lexicon)
                true_text = raw_texts[b_idx].strip()

                # 🟢 CASE INSENSITIVE CHECK
                is_match = (corrected_str.lower() == true_text.lower())

                if is_match:
                    correct_predictions += 1
                total_samples += 1
                sample_print_counter += 1

                if sample_print_counter <= 50:
                    status_icon = "✅" if is_match else "❌"
                    print(f"Sample #{sample_print_counter:03d} {status_icon}")
                    print(f"  ├─ Ground Truth:   '{true_text}'")
                    print(f"  ├─ Model Saw:      '{raw_pred_str}'")
                    print(f"  └─ Lexicon Snapped:'{corrected_str}'")
                    print("-" * 50)

    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    print("=" * 50)
    print(f"📊 TEST DATASET SUMMARY (RESIDUAL CAPACITY):")
    print(f"   🎯 Auto-Corrected Match Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_samples})")


if __name__ == "__main__":
    verify_on_test_data()