import os
import sys
import torch
import torch.nn as nn
import joblib
import cv2
import numpy as np
import pandas as pd

# Dynamically anchor directory structure relative to this file's position
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

try:
    from scripts.medical_detector_cnn import MedicalDetectorCNN  # Your U-Net Architecture
except ImportError:
    from medical_detector_cnn import MedicalDetectorCNN


# ====================================================================
# 1. INTEGRATED RECOGNITION ARCHITECTURE BLOCK
# ====================================================================
class MedicalLabelEncoder:
    def __init__(self):
        self.chars = " %()-./012345678?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.char_to_num = {char: i + 1 for i, char in enumerate(self.chars)}
        self.num_to_char = {i + 1: char for i, char in enumerate(self.chars)}

    def decode(self, nums):
        res = []
        for i, num in enumerate(nums):
            if num != 0 and (i == 0 or num != nums[i - 1]):
                res.append(self.num_to_char.get(num, ""))
        return "".join(res)

    @property
    def vocab_size(self):
        return len(self.chars) + 1


class MedicalCRNN(nn.Module):
    def __init__(self, vocab_size):
        super(MedicalCRNN, self).__init__()
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
        self.hidden_size = 256
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
# 2. CORE PIPELINE IMPLEMENTATION
# ====================================================================
class OCRReaderPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = MedicalLabelEncoder()

        # 1. Load U-Net Segmentation Detector
        self.detector = None
        detector_path = os.path.join(MODEL_DIR, 'medical_detector.pth')
        if os.path.exists(detector_path):
            self.detector = MedicalDetectorCNN(n_channels=1, n_classes=1).to(self.device)
            self.detector.load_state_dict(torch.load(detector_path, map_location=self.device))
            self.detector.eval()
        else:
            print(f"⚠️ Warning: Segmentation weights missing at {detector_path}.")

        # 2. Load Recognition Matrix (MedicalCRNN)
        self.recognizer = None
        recognizer_path = os.path.join(MODEL_DIR, 'MedicalCRNN_v1.pth')
        if os.path.exists(recognizer_path):
            self.recognizer = MedicalCRNN(self.encoder.vocab_size).to(self.device)
            self.recognizer.load_state_dict(torch.load(recognizer_path, map_location=self.device))
            self.recognizer.eval()
            print("🟢 Handwriting recognition model loaded successfully.")
        else:
            print(f"⚠️ Warning: CRNN weight matrix missing at {recognizer_path}.")

        # 3. Load LightGBM Traffic Router Serializations
        self.router = None
        self.vectorizer = None
        router_path = os.path.join(MODEL_DIR, 'MedicalTrafficRouter_v1.pkl')
        vectorizer_path = os.path.join(MODEL_DIR, 'MedicalTrafficRouter_v1_vectorizer.pkl')

        if os.path.exists(router_path) and os.path.exists(vectorizer_path):
            self.router = joblib.load(router_path)
            self.vectorizer = joblib.load(vectorizer_path)
        else:
            print("⚠️ Warning: Traffic Router or Vectorizer binaries missing.")

        print(f"--- ocr_reader_pipeline initialized on {self.device} ---\n")

    def _split_lines_by_projection(self, block_crop):
        h_crop, w_crop = block_crop.shape[:2]

        if h_crop < 80:
            return [block_crop]

        line_crops = []
        if len(block_crop.shape) == 3:
            gray_crop = cv2.cvtColor(block_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray_crop = block_crop.copy()

        if np.mean(gray_crop) > 127:
            _, thresh_crop = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh_crop = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        horizontal_sum = np.sum(thresh_crop, axis=1)
        in_line = False
        start_y = 0

        for idx, row_sum in enumerate(horizontal_sum):
            if not in_line and row_sum > 0:
                in_line = True
                start_y = max(0, idx - 2)
            elif in_line and row_sum == 0:
                in_line = False
                end_y = min(block_crop.shape[0], idx + 2)
                if (end_y - start_y) > 5:
                    line_crops.append(block_crop[start_y:end_y, :])

        if in_line:
            line_crops.append(block_crop[start_y:, :])

        return line_crops if len(line_crops) > 0 else [block_crop]

    def _preprocess_line_for_crnn(self, line_img):
        if len(line_img.shape) == 3:
            gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_img.copy()

        if np.mean(gray) > 127:
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Smart Crop Ink Extraction Bounds
        pts = np.argwhere(processed == 0)
        if len(pts) > 0:
            y_min, x_min = pts.min(axis=0)
            y_max, x_max = pts.max(axis=0)
            if (y_max - y_min) >= 2 and (x_max - x_min) >= 2:
                processed = processed[y_min:y_max + 1, x_min:x_max + 1]

        h_img, w_img = processed.shape[:2]
        if h_img == 0 or w_img == 0:
            processed = np.ones((10, 10), dtype=np.uint8) * 255
            h_img, w_img = 10, 10

        target_w, target_h = 256, 64
        padded_canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255

        # Calculate target dimensions using safe float conversions
        scale = target_h / float(h_img)
        nw = int(w_img * scale)
        nh = target_h

        if nw > target_w:
            scale = target_w / float(w_img)
            nw = target_w
            nh = int(h_img * scale)

        # 🎯 BULLETPROOF GEOMETRIC PROTECTION
        nw = max(1, nw)
        nh = max(1, nh)

        resized = cv2.resize(processed, (nw, nh), interpolation=cv2.INTER_LINEAR)

        start_x = 5
        start_y = max(0, (target_h - nh) // 2)
        actual_w = min(nw, target_w - start_x)

        padded_canvas[start_y:start_y + nh, start_x:start_x + actual_w] = resized[:, :actual_w]

        tensor_img = padded_canvas.astype(np.float32) / 255.0
        tensor_img = (tensor_img - 0.5) / 0.5
        return torch.from_numpy(tensor_img).unsqueeze(0).unsqueeze(0).float().to(self.device)

    def process_image(self, image_input, true_label=None):
        if isinstance(image_input, str):
            raw_img = cv2.imread(image_input)
        else:
            try:
                image_input.seek(0)
                file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
                raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image_input.seek(0)
            except AttributeError:
                file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
                raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if raw_img is None:
            raise ValueError("Pipeline error: Decoded image buffer array is empty.")

        orig_h, orig_w = raw_img.shape[:2]
        gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

        is_full_prescription = True
        aspect_ratio = orig_w / float(orig_h)
        if aspect_ratio > 2.0 or orig_h < 150:
            is_full_prescription = False
            print(f"⚡ Pipeline Gate: Single-word crop validated ({orig_w}x{orig_h}). Bypassing splitter.")

        img_input = cv2.resize(gray_img, (512, 512)) / 255.0
        img_tensor = torch.from_numpy(img_input).unsqueeze(0).unsqueeze(0).float().to(self.device)

        mask = np.zeros((512, 512), dtype=np.uint8)
        if self.detector is not None and is_full_prescription:
            with torch.no_grad():
                mask_output = self.detector(img_tensor)
                mask = (mask_output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

        extracted_line_crops = []

        if is_full_prescription:
            resized_mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (95, 8))
            processed_mask = cv2.morphologyEx(resized_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

            for contour in sorted_contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w < 25 or h < 10:
                    continue

                comp_ratio = w / float(h)
                if 0.8 <= comp_ratio <= 1.3 and w < 140:
                    continue

                padding = 6
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(orig_w, x + w + padding)
                y_end = min(orig_h, y + h + padding)

                block_crop = raw_img[y_start:y_end, x_start:x_end]
                tokenized_lines = self._split_lines_by_projection(block_crop)
                extracted_line_crops.extend(tokenized_lines)
        else:
            extracted_line_crops.append(raw_img)

        print(f"📊 Layout Pipeline Status: Processing {len(extracted_line_crops)} clean segments.")

        decoded_words_list = []
        if self.recognizer is not None and len(extracted_line_crops) > 0:
            for line_crop in extracted_line_crops:
                tensor_input = self._preprocess_line_for_crnn(line_crop)

                with torch.no_grad():
                    num_dirs = 2
                    h0 = torch.zeros(self.recognizer.num_layers * num_dirs, 1, self.recognizer.hidden_size).to(self.device)
                    c0 = torch.zeros(self.recognizer.num_layers * num_dirs, 1, self.recognizer.hidden_size).to(self.device)

                    log_probs = self.recognizer(tensor_input, (h0, c0))
                    preds = log_probs.argmax(dim=2).squeeze(0).cpu().numpy()

                word_text = self.encoder.decode(preds).strip()
                if word_text:
                    decoded_words_list.append(word_text)

            ocr_text_output = " ".join(decoded_words_list)
        else:
            ocr_text_output = ""

        category_label = "Prescription/Symptom"
        confidence_score = 100.0

        if self.router and self.vectorizer and ocr_text_output.strip():
            vec_text = self.vectorizer.transform([ocr_text_output])
            pred_label = self.router.predict(vec_text)[0]
            confidence_score = np.max(self.router.predict_proba(vec_text)) * 100
            category_label = "Prescription/Symptom" if pred_label == 0 else "Lab Report"

        accuracy = None
        if true_label is not None and self.router:
            accuracy = 100.0 if pred_label == true_label else 0.0

        return {
            "ocr_text": ocr_text_output,
            "category": category_label,
            "confidence": f"{confidence_score:.2f}%",
            "router_accuracy": accuracy,
            "mask_preview": mask,
            "line_crops_list": extracted_line_crops
        }


if __name__ == "__main__":
    pipeline = OCRReaderPipeline()
    test_file = "sample_test.png"

    if os.path.exists(test_file):
        results = pipeline.process_image(test_file, true_label=0)
        print("\n--- Core Pipeline Execution Diagnostics ---")
        print(f"Extracted Active Text Matrix: '{results['ocr_text']}'")
        print(f"Document Classification Rule: {results['category']}")
        print(f"Router Confidence Score:      {results['confidence']}")
    else:
        print(f"\n--- Script ready. Place '{test_file}' in root folder to run validation tracing locally. ---")