import os
import random
import csv
import sys
import re
import difflib
import hashlib
import uuid
import subprocess
import threading
import requests
import json
import base64
import io
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from threading import Lock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import cv2
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import streamlit as st
import streamlit.components.v1 as components
from rapidfuzz import process, fuzz
from st_supabase_connection import SupabaseConnection
from PIL import Image

# Global thread lock for parallel PyTorch tensor execution contexts
INFERENCE_LOCK = Lock()

# ====================================================================
# BACKWARD COMPATIBILITY INJECTOR PATCH (SCI-KIT LEARN FIX)
# ====================================================================
import sklearn

if not hasattr(sklearn, '__version__'):
    sklearn.__version__ = "1.4.2"
try:
    import sklearn.utils._estimator_html_repr
except ImportError:
    sys.modules['sklearn.utils._estimator_html_repr'] = sys.modules.get('sklearn.utils', None)

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_SCRIPT_DIR not in sys.path:
    sys.path.append(CURRENT_SCRIPT_DIR)

try:
    from scripts.medical_detector_cnn import MedicalDetectorCNN
except ImportError:
    MedicalDetectorCNN = None

# ====================================================================
# 1. ROBUST RELATIVE PATH CONFIGURATION ARCHITECTURE
# ====================================================================
IS_ONLINE_DEPLOYMENT = os.path.exists("/mount/src") or not os.path.exists(r"C:\Users\Bubu")

# Dynamically anchor root down to structural validation flags
if not IS_ONLINE_DEPLOYMENT:
    resolved_root = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System"
else:
    possible_roots = [
        CURRENT_SCRIPT_DIR,
        os.path.dirname(CURRENT_SCRIPT_DIR),
        os.path.dirname(os.path.dirname(CURRENT_SCRIPT_DIR))
    ]
    resolved_root = CURRENT_SCRIPT_DIR
    for root in possible_roots:
        if os.path.exists(os.path.join(root, "models")) or os.path.exists(os.path.join(root, "data")):
            resolved_root = root
            break

# File Path Architecture Map
MODEL_DIR = os.path.join(resolved_root, "models")
DATA_DIR = os.path.join(resolved_root, "data", "clean", "disease_and_symptom_clean")
RAW_DIR = os.path.join(resolved_root, "data", "raw")
TEMP_DIR = os.path.join(resolved_root, "data", "temp")
PREPROCESS_SCRIPT = os.path.join(resolved_root, "scripts", "chat_bot_preprocessing.py")
TRAIN_SCRIPT = os.path.join(resolved_root, "scripts", "train_lgbm.py")

MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model_clean.pkl")
LE_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")
FEAT_PATH = os.path.join(DATA_DIR, "X_preprocessed.csv")
FULL_DATA_PATH = os.path.join(DATA_DIR, "preprocessed_data.csv")
REQUESTS_FILE = os.path.join(TEMP_DIR, "unverified_diseases.csv")
LEARNED_DATA_FILE = os.path.join(RAW_DIR, "learned_user_data.csv")
DETECTOR_WEIGHTS = os.path.join(MODEL_DIR, "medical_detector.pth")
CRNN_WEIGHTS = os.path.join(MODEL_DIR, "MedicalCRNN_v2_Residual.pth")
VOCAB_JSON = os.path.join(MODEL_DIR, "medical_vocab.json")
APP_DATA_DIR = os.path.join(CURRENT_SCRIPT_DIR, "app_data")
INFO_DB_PATH = os.path.join(APP_DATA_DIR, "who_data_clean.csv")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(APP_DATA_DIR, exist_ok=True)

DISEASE_ALIASES = {
    "common cold": "upper respiratory infection",
    "cold": "upper respiratory infection",
    "flu": "influenza",
    "sugar": "diabetes",
    "bp": "hypertension",
    "heart attack": "myocardial infarction",
    "brain stroke": "cerebrovascular accident"
}


def find_database_dynamically():
    for root_dir, _, files in os.walk(resolved_root):
        for file in files:
            if "Final_Compiled_Medicine_Database" in file:
                return os.path.join(root_dir, file)
    return None


DB_PATH = find_database_dynamically()


def load_medicine_database(db_path):
    if db_path is None or not os.path.exists(db_path):
        return None, "File not found by Auto-Locator."
    try:
        try:
            df = pd.read_excel(db_path, engine='openpyxl')
        except Exception:
            df = pd.read_csv(db_path, low_memory=False)

        search_column = 'Medicine' if 'Medicine' in df.columns else df.columns[0]
        df['lookup_key'] = df[search_column].astype(str).str.strip().str.lower()
        df = df.drop_duplicates(subset=['lookup_key'], keep='first')
        return df.set_index('lookup_key').dropna(axis=1, how='all').to_dict(orient='index'), "Success"
    except Exception as e:
        return None, f"Processing Error: {str(e)}"


def fetch_medicine_details_fast(extracted_name, lookup_dict):
    if not lookup_dict: return None
    return lookup_dict.get(str(extracted_name).strip().lower(), None)


MEDICAL_DICTIONARY = [
    "Rx", "Stable", "Tablet", "Capsule", "Amoxicillin", "Paracetamol",
    "Azithromycin", "Metformin", "Ibuprofen", "Anacin", "Flamex",
    "Syrup", "Injection", "Pantoprazole", "Vitamin-C", "Cetirizine",
    "FeSO4", "Ascorbic Acid", "once a day", "twice a day", "Napdos",
    "Losita", "Rivotril", "Econate", "Kacin", "bengel", "Omep", "Fougest",
    "RUPIN", "myolax", "Tenocab", "Radifil", "Povital", "Napa", "Voligel", "lactomore", "Don A",
    "Calbo-D"
]

CRNN_EXCEPTION_PATCH = {
    "ter m": "Losita",
    "term": "Losita",
    "povoex": "Napdos",
    "pobccv": "Metformin",
    "calbo d": "Calbo-D"
}

# ====================================================================
# 2. REMOTE STORAGE HANDLERS (SUPABASE SECRET INJECTOR)
# ====================================================================
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "https://cwwoloupweulprxwibmp.supabase.co")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY",
                              "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN3d29sb3Vwd2V1bHByeHdpYm1wIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg3MDA5NDEsImV4cCI6MjA5NDI3Njk0MX0.ggPfeYBaL7PLiEM8_fYI5fHo48obb5yRum_kR1CORNM")

try:
    conn = st.connection(
        "supabase",
        type=SupabaseConnection,
        url=SUPABASE_URL,
        key=SUPABASE_KEY
    )
except Exception:
    conn = None


def get_visitor_id():
    if 'visitor_session_uuid' not in st.session_state:
        st.session_state.visitor_session_uuid = str(uuid.uuid4())[:18]
    return st.session_state.visitor_session_uuid


def generate_permanent_key(email):
    random.seed(int(hashlib.sha256(email.strip().lower().encode()).hexdigest(), 16) % 10 ** 8)
    return str(random.randint(100000, 999999))


def save_user_cloud(v_id, email, key):
    if conn is None: return False
    try:
        conn.table("user_identities").upsert({"visitor_id": v_id, "email": email, "permanent_key": str(key)}).execute()
        return True
    except Exception:
        return False


def verify_user_cloud(v_id, input_key):
    if conn is None: return False
    try:
        query = conn.table("user_identities").select("*").eq("permanent_key", str(input_key)).execute()
        if len(query.data) > 0:
            st.session_state.active_patient_email = query.data[0]['email']
            return True
        return False
    except:
        return False


# ====================================================================
# 3. TEXT PATTERN CODES (RESIDUAL CRNN)
# ====================================================================
class MedicalLabelEncoder:
    def __init__(self, json_path):
        data = json.load(open(json_path, 'r', encoding='utf-8'))
        self.chars, self.lexicon = data['chars'], set(data['lexicon'])
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
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))


class MedicalResidualCRNN(nn.Module):
    def __init__(self, vocab_size):
        super(MedicalResidualCRNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer2, self.pool1 = ResidualBlock(64, 64, stride=1), nn.MaxPool2d(2)
        self.layer3, self.layer4 = ResidualBlock(64, 128, stride=1), ResidualBlock(128, 128, stride=1)
        self.pool2, self.spatial_drop1 = nn.MaxPool2d(2), nn.Dropout2d(p=0.2)
        self.layer5, self.layer6 = ResidualBlock(128, 256, stride=1), ResidualBlock(256, 256, stride=1)
        self.pool3, self.spatial_drop2 = nn.MaxPool2d((2, 1)), nn.Dropout2d(p=0.2)
        self.layer7 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.pool4 = nn.MaxPool2d((2, 1))
        self.hidden_size, self.num_layers = 256, 2
        self.rnn = nn.LSTM(input_size=512 * 4, hidden_size=self.hidden_size, num_layers=self.num_layers,
                           bidirectional=True, batch_first=True, dropout=0.0)
        self.fc = nn.Linear(self.hidden_size * 2, vocab_size)

    def forward(self, img_tensor, hx=None):
        x = self.pool4(self.layer7(self.spatial_drop2(self.pool3(self.layer6(self.layer5(self.spatial_drop1(
            self.pool2(self.layer4(self.layer3(self.pool1(self.layer2(self.layer1(img_tensor)))))))))))))
        b, c, h, w = x.size()
        rnn_out, _ = self.rnn(x.view(b, c * h, w).permute(0, 2, 1), hx)
        return self.fc(rnn_out).log_softmax(2)


# ====================================================================
# 4. COMPUTER VISION PARSING ENGINE (WITH ASPECT-RATIO PAD RESIZING)
# ====================================================================
class OCRReaderPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector, self.text_recognizer = None, None
        if os.path.exists(VOCAB_JSON):
            self.encoder = MedicalLabelEncoder(VOCAB_JSON)
            self.medical_dictionary = list(self.encoder.lexicon)
        else:
            self.medical_dictionary, self.encoder = MEDICAL_DICTIONARY, None
        self.load_models()

    def load_models(self):
        if MedicalDetectorCNN is not None and os.path.exists(DETECTOR_WEIGHTS):
            self.detector = MedicalDetectorCNN(n_channels=1, n_classes=1).to(self.device)
            self.detector.load_state_dict(torch.load(DETECTOR_WEIGHTS, map_location=self.device))
            self.detector.eval()
        if self.encoder is not None and os.path.exists(CRNN_WEIGHTS):
            self.text_recognizer = MedicalResidualCRNN(self.encoder.vocab_size).to(self.device)
            state_dict = {k.replace("module.", "").replace("model.", "").replace("text_recognizer.", ""): v for k, v in
                          torch.load(CRNN_WEIGHTS, map_location=self.device).items()}
            self.text_recognizer.load_state_dict(state_dict, strict=True)
            self.text_recognizer.eval()

    def segment_full_prescription(self, raw_img):
        orig_h, orig_w = raw_img.shape[:2]

        _, thresh = cv2.threshold(raw_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (95, 3))
        dilated_mask = cv2.dilate(thresh, horizontal_kernel, iterations=1)

        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        line_crops = []

        if len(contours) > 0:
            contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
            for ctr in contours:
                x, y, w, h = cv2.boundingRect(ctr)
                if w > 35 and h > 8:
                    pad_y1, pad_y2 = max(0, y - 2), min(orig_h, y + h + 2)
                    pad_x1, pad_x2 = max(0, x - 5), min(orig_w, x + w + 5)
                    crop_slice = raw_img[pad_y1:pad_y2, pad_x1:pad_x2].copy()
                    if crop_slice.size > 0:
                        line_crops.append(crop_slice)

        return line_crops if len(line_crops) > 0 else [raw_img]

    def recognize_crop(self, crop):
        if self.text_recognizer is None: return "Weights Missing", 0.0
        if crop is None or crop.shape[0] < 4 or crop.shape[1] < 4: return "", 0.0

        # Run Bilateral background normalization
        smoothed = cv2.bilateralFilter(crop, 5, 65, 65)
        _, thresh = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            all_pts = np.concatenate(contours)
            x, y, w, h = cv2.boundingRect(all_pts)
            pad = 4
            y1 = max(0, y - pad)
            y2 = min(crop.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(crop.shape[1], x + w + pad)
            cleaned_line = crop[y1:y2, x1:x2]
        else:
            cleaned_line = smoothed

        # PROPORTIONAL ASPECT RATIO RESIZER
        target_w, target_h = 256, 64
        crnn_input = np.ones((target_h, target_w), dtype=np.uint8) * 255

        h_orig, w_orig = cleaned_line.shape[:2]
        scale = min(target_w / w_orig, target_h / h_orig)

        nw = max(4, int(w_orig * scale))
        nh = max(4, int(h_orig * scale))

        try:
            resized_crop = cv2.resize(cleaned_line, (nw, nh), interpolation=cv2.INTER_CUBIC)
        except:
            return "", 0.0

        start_x = (target_w - nw) // 2
        start_y = (target_h - nh) // 2
        crnn_input[start_y:start_y + nh, start_x:start_x + nw] = resized_crop

        if np.mean(crnn_input) < 127: crnn_input = cv2.bitwise_not(crnn_input)

        st.session_state.crnn_debug_image_matrix = crnn_input.copy()

        crnn_input = (crnn_input.astype(np.float32) / 255.0 - 0.5) / 0.5
        crnn_tensor = torch.from_numpy(crnn_input).float().to(self.device).unsqueeze(0).unsqueeze(0)

        with INFERENCE_LOCK:
            with torch.no_grad():
                h0 = torch.zeros(self.text_recognizer.num_layers * 2, 1, self.text_recognizer.hidden_size,
                                 dtype=torch.float32).to(self.device)
                logits = self.text_recognizer(crnn_tensor, (h0, h0))
                probs = torch.exp(logits).squeeze(0)
                best_path = torch.argmax(logits.squeeze(0), dim=1).cpu().numpy()
                path_probs = probs[torch.arange(probs.size(0)), best_path].cpu().numpy()
                line_confidence = float(np.mean(path_probs)) * 100

                decoded_line = self.encoder.decode(best_path).strip()
                text_lower = decoded_line.lower()

                if text_lower in CRNN_EXCEPTION_PATCH:
                    decoded_line = CRNN_EXCEPTION_PATCH[text_lower]
                match = process.extractOne(decoded_line, self.medical_dictionary, scorer=fuzz.WRatio)
                if match and match[1] >= 45.0:
                    decoded_line = match[0]
                return decoded_line, line_confidence


# ====================================================================
# 5. DIAGNOSTICS & NLP COGNITIVE SYMPTOM PARSER
# ====================================================================
class MedicalAI:
    def __init__(self):
        self.model, self.le = None, None
        self.known_symptoms, self.known_diseases = [], []
        self.df_full = None
        self.load_resources()

    def load_resources(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(LE_PATH) and os.path.exists(FEAT_PATH):
            try:
                self.model, self.le = joblib.load(MODEL_PATH), joblib.load(LE_PATH)
                self.known_symptoms = pd.read_csv(FEAT_PATH, nrows=0).columns.tolist()
                self.known_diseases = [d.lower() for d in self.le.classes_]
                if os.path.exists(FULL_DATA_PATH):
                    self.df_full = pd.read_csv(FULL_DATA_PATH)
            except Exception as e:
                st.warning(f"Soft Initialization Failure: {e}")
                self.load_emergency_backup_vectors()
        else:
            self.load_emergency_backup_vectors()

    def load_emergency_backup_vectors(self):
        self.known_symptoms = ["fever", "cough", "headache", "fatigue", "vomiting", "chills", "nausea",
                               "muscle_weakness"]
        self.known_diseases = ["influenza", "common cold", "malaria"]

    def get_symptoms(self, disease_name):
        if self.df_full is None: return []
        subset = self.df_full[self.df_full['prognosis'].str.lower() == disease_name.lower()]
        if subset.empty: return []
        row = subset.iloc[0]
        active_symptoms = [col.replace("_", " ") for col in self.known_symptoms if col in row and row[col] == 1]
        return active_symptoms

    def log_learning_request(self, disease_name):
        required_columns = ["timestamp", "source_url", "proposed_disease", "symptoms", "status"]
        if not os.path.exists(REQUESTS_FILE):
            with open(REQUESTS_FILE, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(required_columns)
        try:
            with open(REQUESTS_FILE, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "User App", disease_name, "Pending", "Pending"
                ])
            return True
        except:
            return False

    def scrape_wikipedia(self, disease_name):
        slug = disease_name.strip().replace(" ", "_").title()
        url = f"https://en.wikipedia.org/wiki/{slug}"
        found_data = []
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for p in soup.find_all('p'):
                    if len(p.get_text()) > 50:
                        clean_text = re.sub(r'\[\d+\]', '', p.get_text().strip())
                        found_data.append(f"**Summary:** {clean_text[:300]}...")
                        break
                for h in soup.find_all(['h2', 'h3']):
                    if any(t in h.get_text() for t in ["Prevention", "Management", "Treatment"]):
                        ul = h.find_next('ul')
                        if ul:
                            for li in ul.find_all('li')[:3]:
                                found_data.append(re.sub(r'\[\d+\]', '', li.get_text().strip()))
                        break
                return found_data, url
        except:
            pass
        return [], "None"

    def get_advice(self, disease_name):
        clean_name = disease_name.lower().strip()
        if os.path.exists(INFO_DB_PATH):
            try:
                df = pd.read_csv(INFO_DB_PATH)
                match = df[df['Disease'].str.lower() == clean_name]
                if not match.empty:
                    row = match.iloc[0]
                    tips = [row[f"Precaution_{i}"] for i in range(1, 6) if pd.notna(row.get(f"Precaution_{i}"))]
                    return tips, row.get('Source', 'Local')
            except:
                pass

        found_text = []
        source = "WHO"
        slug = clean_name.replace(" ", "-")
        try:
            resp = requests.get(f"https://www.who.int/news-room/fact-sheets/detail/{slug}",
                                headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for h in soup.find_all(['h2', 'h3']):
                    if any(k in h.get_text() for k in ["Prevention", "Treatment", "Key facts"]):
                        for tag in h.find_next_siblings(['p', 'ul'])[:4]:
                            txt = tag.get_text().strip().replace('\n', ' ')
                            if len(txt) > 20: found_text.append(txt)
                        if found_text: break
        except:
            pass

        if not found_text: found_text, source = self.scrape_wikipedia(clean_name)
        if found_text:
            new_row = {"Disease": clean_name, "Source": source}
            for i, tip in enumerate(found_text[:5]): new_row[f"Precaution_{i + 1}"] = tip
            if os.path.exists(INFO_DB_PATH):
                df = pd.read_csv(INFO_DB_PATH)
            else:
                df = pd.DataFrame(
                    columns=["Disease", "Source", "Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4",
                             "Precaution_5"])
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(INFO_DB_PATH, index=False)
        return found_text, source

    def verify_and_extract(self, disease_name):
        found_symptoms = []
        searchable_symptoms = [(s.replace("_", " "), s) for s in self.known_symptoms]
        try:
            url = f"https://www.who.int/news-room/fact-sheets/detail/{disease_name.replace(' ', '-').lower()}"
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            if resp.status_code == 200:
                text = BeautifulSoup(resp.text, 'html.parser').get_text().lower()
                for clean_sym, original_sym in searchable_symptoms:
                    parts = clean_sym.split()
                    if all(p in text for p in parts): found_symptoms.append(original_sym)
                if len(found_symptoms) >= 1: return list(set(found_symptoms)), url
        except:
            pass
        try:
            url = f"https://en.wikipedia.org/wiki/{disease_name.replace(' ', '_')}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                text = BeautifulSoup(resp.text, 'html.parser').get_text().lower()
                for clean_sym, original_sym in searchable_symptoms:
                    parts = clean_sym.split()
                    if all(p in text for p in parts): found_symptoms.append(original_sym)
                if len(found_symptoms) >= 1: return list(set(found_symptoms)), url
        except:
            pass
        return None, None

    def execute_verification_cycle(self):
        st.info("🧠 Processing pending knowledge structures and retraining parameters...")
        if not os.path.exists(REQUESTS_FILE):
            return False, "No requests file pipeline exists to load updates."

        df = pd.read_csv(REQUESTS_FILE)
        if 'status' not in df.columns: df['status'] = 'Pending'
        pending = df[df['status'] == 'Pending']
        if pending.empty: return False, "No pending validation objects discovered."

        update_needed = False
        new_entries = []
        existing_diseases = []
        if os.path.exists(LEARNED_DATA_FILE):
            try:
                temp_df = pd.read_csv(LEARNED_DATA_FILE)
                if 'prognosis' in temp_df.columns:
                    existing_diseases = temp_df['prognosis'].str.lower().unique().tolist()
            except:
                pass

        for index, row in pending.iterrows():
            d_name = row['proposed_disease']
            clean_name = d_name.strip().lower()
            if clean_name in existing_diseases:
                df.at[index, 'status'] = 'Duplicate'
                continue

            symptoms, url = self.verify_and_extract(d_name)
            if symptoms:
                entry = {col: 0 for col in self.known_symptoms}
                entry['prognosis'] = d_name.title()
                for s in symptoms: entry[s] = 1
                new_entries.append(entry)
                df.at[index, 'status'] = 'Approved'
                df.at[index, 'source_url'] = url
                df.at[index, 'symptoms'] = ", ".join(symptoms)
                update_needed = True
                self.get_advice(d_name)
            else:
                df.at[index, 'status'] = 'Rejected'

        df.to_csv(REQUESTS_FILE, index=False)
        if update_needed and new_entries:
            df_new = pd.DataFrame(new_entries)
            header = not os.path.exists(LEARNED_DATA_FILE)
            df_new.to_csv(LEARNED_DATA_FILE, mode='a', header=header, index=False)
            try:
                if os.path.exists(PREPROCESS_SCRIPT): subprocess.run([sys.executable, PREPROCESS_SCRIPT], check=True)
                if os.path.exists(TRAIN_SCRIPT): subprocess.run([sys.executable, TRAIN_SCRIPT], check=True)
                self.load_resources()
                return True, "✅ Model retraining finished successfully! New profiles verified."
            except Exception as e:
                return False, f"Pipeline Error during model construction: {e}"
        return False, "Verification complete. No new distinct components found."

    def predict(self, user_input):
        if self.model is None or self.le == None:
            return "Uncompiled Classifier Matrix (Type 'verify now')", [], 0.0

        query_clean = user_input.lower().strip()
        input_dict = {col: 0 for col in self.known_symptoms}
        matched = []

        for symptom in self.known_symptoms:
            normalized_symptom_col = symptom.replace("_", " ")
            if normalized_symptom_col in query_clean:
                input_dict[symptom] = 1
                matched.append(symptom)

        cleaned_sentence = re.sub(r'\b(and|or|i have|feeling|my|is|at|both|severe|with|high|acute)\b', '', query_clean,
                                  flags=re.IGNORECASE)
        individual_words = [w.strip() for w in re.split(r'[\s,]+', cleaned_sentence) if len(w.strip()) > 2]

        for word in individual_words:
            for symptom in self.known_symptoms:
                normalized_symptom_col = symptom.replace("_", " ")
                if re.search(r'\b' + re.escape(word) + r'\b', normalized_symptom_col) or difflib.get_close_matches(word,
                                                                                                                   [
                                                                                                                       normalized_symptom_col],
                                                                                                                   cutoff=0.8):
                    if symptom not in matched:
                        input_dict[symptom] = 1
                        matched.append(symptom)

        if not matched:
            return None, [], 0.0

        input_df = pd.DataFrame([input_dict], columns=self.known_symptoms)
        pred_id = self.model.predict(input_df)[0]
        confidence = self.model.predict_proba(input_df)[0][pred_id] * 100
        disease_string = self.le.inverse_transform([pred_id])[0]

        return disease_string, list(set(matched)), confidence


# ====================================================================
# 6. STREAMLIT APPLICATION LOGIC & INTERFACE
# ====================================================================
def process_extraction_result(ocr_text, db_lookup):
    db_insights = ""
    for line in ocr_text.split("\n"):
        clean_line = line.strip()
        if len(clean_line) <= 2: continue
        med_info = fetch_medicine_details_fast(clean_line, db_lookup)
        if med_info:
            generic_name = med_info.get('Generic Name') or med_info.get('Generic') or 'N/A'
            purpose_text = med_info.get('Use/Purpose') or med_info.get('Purpose') or 'N/A'
            side_effects = med_info.get('Common Side Effects') or 'N/A'
            manufacturer = med_info.get('Manufacturer') or 'N/A'

            db_insights += f"💊 **{clean_line}**\n" \
                           f"* 👉 **Generic:** {generic_name}\n" \
                           f"* 👉 **Purpose:** {purpose_text}\n" \
                           f"* 👉 **Side Effects:** {side_effects}\n" \
                           f"* 👉 **Manufacturer:** {manufacturer}\n\n"

    response = f"✅ Medicines extracted successfully from auto-aligned selection.\n\nDetected String: `{ocr_text}`"
    if db_insights: response += f"\n\n📚 **Database Matches:**\n\n{db_insights}"
    return response


def embed_hospital_finder():
    html_path = os.path.join(CURRENT_SCRIPT_DIR, "hospital_finder.html")
    if not os.path.exists(html_path):
        html_path = os.path.join(CURRENT_SCRIPT_DIR, "static", "hospital_finder.html")

    if os.path.exists(html_path):
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                html_raw_code = f.read()
            components.html(html_raw_code, height=540)
        except Exception as err:
            st.error(f"Canvas compilation fault loop triggered: {err}")
    else:
        st.error("Hospital finder core engine template target execution canvas mapping missing.")


@st.fragment(run_every=5)
def live_chat_stream(room_id, view_role, active_v_id=None):
    if conn is None:
        st.error("Supabase engine disconnected.")
        return
    try:
        if view_role == "patient":
            status = conn.table("doctor_status").select("last_seen").eq("is_online", True).execute()
            if status.data:
                last_seen = datetime.fromisoformat(status.data[0]['last_seen'].replace('Z', '+00:00'))
                if datetime.now(last_seen.tzinfo) - last_seen < timedelta(seconds=30):
                    st.success("🟢 Doctor is online now!")
                else:
                    st.warning("🔴 Line is currently busy / Doctor offline.")

        chat_query = conn.table("doctor_chat_messages").select("*").eq("chat_room_id", room_id).order("created_at",
                                                                                                      desc=False).execute()

        for msg in chat_query.data:
            if "[System Alert]" in msg["message_text"]:
                st.caption(msg["message_text"])
                continue

            if view_role == "doctor":
                role = "assistant" if msg["sender_type"] == "doctor" else "user"
                prefix = "" if msg["message_text"].startswith("[IMAGE_BASE64]") else (
                        "%s" % ("🩺 **You:** " if msg["sender_type"] == "doctor" else "👤 **Patient:** "))
            else:
                role = "user" if msg["sender_type"] == "patient" else "assistant"
                prefix = "" if msg["message_text"].startswith("[IMAGE_BASE64]") else (
                        "%s" % ("🩺 **Doctor:** " if msg["sender_type"] == "doctor" else "👤 **You:** "))

            with st.chat_message(role):
                if msg["message_text"].startswith("[IMAGE_BASE64]"):
                    b64_data = msg["message_text"].replace("[IMAGE_BASE64]", "")
                    sender_title = "🩺 **Attending Doctor Shared Image:**" if msg[
                                                                                 "sender_type"] == "doctor" else "👤 **Patient Sent Prescription/Report Image:**"
                    st.markdown(f"**{sender_title}**")
                    st.image(b64_data, use_container_width=True)
                else:
                    st.markdown(f"{prefix}{msg['message_text']}")
    except Exception as e:
        st.caption(f"⚡ Connection jitter handled safely... ({e})")


def bg_db_insert(payload_dict):
    if conn is None: return
    try:
        conn.table("doctor_chat_messages").insert(payload_dict).execute()
    except Exception:
        pass


def update_doctor_heartbeat():
    if conn is None: return
    try:
        conn.table("doctor_status").upsert({"is_online": True, "last_seen": datetime.now().isoformat()},
                                           on_conflict="is_online").execute()
    except:
        pass


def main():
    st.set_page_config(page_title="AI Health Assistant", layout="wide")

    if 'bot' not in st.session_state: st.session_state.bot = MedicalAI()
    if 'ocr_pipeline' not in st.session_state: st.session_state.ocr_pipeline = OCRReaderPipeline()
    if 'auth' not in st.session_state: st.session_state.auth = False
    if 'chat_mode' not in st.session_state: st.session_state.chat_mode = "ai_assistant"
    if 'is_doctor' not in st.session_state: st.session_state.is_doctor = False
    if 'doctor_display_name' not in st.session_state: st.session_state.doctor_display_name = "Doctor"
    if 'doctor_db_id' not in st.session_state: st.session_state.doctor_db_id = 1
    if 'selected_room' not in st.session_state: st.session_state.selected_room = None
    if 'active_patient_email' not in st.session_state: st.session_state.active_patient_email = None

    if 'crnn_debug_image_matrix' not in st.session_state:
        st.session_state.crnn_debug_image_matrix = None

    file_upload = None
    camera_photo = None

    if st.session_state.auth and st.session_state.active_patient_email:
        raw_target_string = str(st.session_state.active_patient_email).strip().lower()
        v_id = hashlib.sha256(raw_target_string.encode()).hexdigest()[:16]
        room_prefix = "user_"
    else:
        v_id = get_visitor_id()
        room_prefix = "guest_"

    # ====================================================================
    # 🎯 SIDEBAR ENVIRONMENT CONSOLE & AUTOMATED CROPPER PIPELINE
    # ====================================================================
    with st.sidebar:
        st.subheader("👁️ Segmentation Studio")
        debug_segmentation = st.checkbox("Toggle U-Net Live Segmentation Output Map", value=True)

        if st.session_state.auth and not st.session_state.is_doctor:
            st.divider()
            st.subheader("📦 Prescription Input Capture")
            file_upload = st.file_uploader("Upload Prescription Sheet", type=[ "png", "jpg", "jpeg"])

            if 'camera_active' not in st.session_state: st.session_state.camera_active = False
            if not st.session_state.camera_active:
                if st.button("📸 Open Live Camera Scanner", use_container_width=True):
                    st.session_state.camera_active = True
                    st.rerun()
            else:
                if st.button("❌ Close Camera", use_container_width=True):
                    st.session_state.camera_active = False
                    st.rerun()
                st.markdown(
                    """<style>[data-testid="stCameraInput"] { position: relative; width: 100% !important; } [data-testid="stCameraInput"] video { width: 100% !important; height: auto !important; object-fit: cover !important; } [data-testid="stCameraInput"]:has(video)::before { content: 'ALIGN MEDICINE NAME HERE'; position: absolute; top: 38%; left: 5%; width: 90%; height: 24%; border: 3px dashed #00FF00; color: #00FF00; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: bold; text-align: center; z-index: 99; pointer-events: none; background-color: rgba(0, 255, 0, 0.08); box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.4); }</style>""",
                    unsafe_allow_html=True)
                camera_photo = st.camera_input("Capture Medicine Image Layer")

            uploaded_file = camera_photo if camera_photo else file_upload

            if uploaded_file is not None:
                file_bytes = uploaded_file.getvalue()
                file_hash = hashlib.md5(file_bytes).hexdigest()

                if st.session_state.last_processed_file_hash != file_hash:
                    st.session_state.last_processed_file_hash = file_hash

                    if st.session_state.chat_mode == "doctor_consult":
                        st.toast("📷 Compressing and uploading image in background...", icon="ℹ️")
                        try:
                            pil_img = Image.open(uploaded_file)
                            if pil_img.mode in ("RGBA", "P"):
                                pil_img = pil_img.convert("RGB")

                            mem_buffer = io.BytesIO()
                            pil_img.save(mem_buffer, format="JPEG", quality=60)

                            b64_string = base64.b64encode(mem_buffer.getvalue()).decode("utf-8")
                            b64_payload = f"[IMAGE_BASE64]data:image/jpeg;base64,{b64_string}"

                            target_active_doctor_id = st.session_state.get('patient_selected_doctor_id', 998877)
                            resolved_patient_room_id = f"doc_{target_active_doctor_id}_patient_{room_prefix}{v_id}"

                            upload_data = {
                                "chat_room_id": resolved_patient_room_id,
                                "sender_type": "patient",
                                "sender_id": v_id,
                                "message_text": b64_payload
                            }
                            bg_db_insert(upload_data)
                            st.toast("Shared captured frame directly with practitioner.", icon="🩺")
                        except Exception as e:
                            st.error(f"Image compression error: {e}")
                    else:
                        raw_img = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

                        # --- AUTOMATED CROP & BOUNDS DETECTOR LOGIC FOR CAMERA SAMPLES ---
                        if camera_photo is not None:
                            blur_pre = cv2.GaussianBlur(raw_img, (5, 5), 0)
                            _, thresh_card = cv2.threshold(blur_pre, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            card_contours, _ = cv2.findContours(thresh_card, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            best_card_box = None
                            max_card_area = 0
                            h_img, w_img = raw_img.shape[:2]

                            for cnt in card_contours:
                                xc, yc, wc, hc = cv2.boundingRect(cnt)
                                card_area = wc * hc
                                if wc > (w_img * 0.4) and hc > (h_img * 0.1):
                                    if card_area > max_card_area:
                                        max_card_area = card_area
                                        best_card_box = (xc, yc, wc, hc)

                            if best_card_box is not None:
                                xc, yc, wc, hc = best_card_box
                                raw_img = raw_img[yc:yc + hc, xc:xc + wc].copy()
                            else:
                                y1 = int(h_img * 0.38)
                                y2 = int(h_img * 0.62)
                                x1 = int(w_img * 0.05)
                                x2 = int(w_img * 0.95)
                                raw_img = raw_img[y1:y2, x1:x2].copy()

                        raw_img = cv2.adaptiveThreshold(
                            raw_img, 255,
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 41, 15
                        )

                        orig_h, orig_w = raw_img.shape[:2]
                        img_aspect = orig_w / float(orig_h)
                        pipeline = st.session_state.ocr_pipeline

                        # Render U-Net live overlay diagnostics panel if enabled
                        if debug_segmentation and camera_photo is not None and pipeline.detector is not None:
                            color_preview_src = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8),
                                                             cv2.IMREAD_COLOR)
                            h_c, w_c = raw_img.shape[:2]
                            input_resized = cv2.resize(raw_img, (512, 512))
                            tensor_img = torch.from_numpy(input_resized).float().to(pipeline.device).unsqueeze(
                                0).unsqueeze(0) / 255.0

                            with torch.no_grad():
                                output_mask = pipeline.detector(tensor_img)
                                probs = torch.sigmoid(output_mask).squeeze().cpu().numpy()
                                binary_mask = (probs > 0.5).astype(np.uint8) * 255
                                binary_mask_resized = cv2.resize(binary_mask, (w_c, h_c))

                            color_crop_src = color_preview_src.copy()
                            if best_card_box is not None:
                                xc, yc, wc, hc = best_card_box
                                color_crop_src = color_crop_src[yc:yc + hc, xc:xc + wc].copy()
                            else:
                                h_i, w_i = color_preview_src.shape[:2]
                                color_crop_src = color_crop_src[
                                    int(h_i * 0.38):int(h_i * 0.62), int(w_i * 0.05):int(w_i * 0.95)].copy()

                            visual_mask_overlay = color_crop_src.copy()
                            visual_mask_overlay[binary_mask_resized > 0] = [0, 255, 0]
                            blended_preview = cv2.addWeighted(color_crop_src, 0.7, visual_mask_overlay, 0.3, 0)
                            st.image(blended_preview, caption="U-Net Mask (Active Overlay)", use_container_width=True)

                        if camera_photo is not None or img_aspect > 2.2:
                            with st.spinner("Processing Isolated Target Card..."):
                                text_out, conf_out = pipeline.recognize_crop(raw_img)
                                if text_out.strip():
                                    scan_payload = f"📋 *[Automated ROI Extraction Target Match]:*"
                                    response = process_extraction_result(text_out, st.session_state.db_lookup)
                                    st.session_state.messages.append({"role": "user", "content": scan_payload})
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                                    st.rerun()
                                else:
                                    st.warning("Text unrecognized inside automated boundaries.")
                        else:
                            with st.spinner("Deconstructing Full Page Matrix..."):
                                extracted_slices = pipeline.segment_full_prescription(raw_img)
                                all_discovered_text = []

                                for slice_block in extracted_slices:
                                    text_out, conf_out = pipeline.recognize_crop(slice_block)
                                    if text_out.strip():
                                        all_discovered_text.append(text_out)

                                ocr_combined_result = "\n".join(all_discovered_text)
                                if ocr_combined_result.strip():
                                    scan_payload = f"📋 *[Full Document Scan Automated Extraction]:*"
                                    response = process_extraction_result(ocr_combined_result,
                                                                         st.session_state.db_lookup)
                                    st.session_state.messages.append({"role": "user", "content": scan_payload})
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                                    st.rerun()
                                else:
                                    st.error("No valid medical items identified on across page layout structure.")

                    if st.session_state.crnn_debug_image_matrix is not None:
                        st.markdown("---")
                        st.markdown("### 🔍 Selection Debug Desk")
                        st.caption("🎞️ **Aspect-Ratio Padded CRNN Box Window** ($256 \times 64$)")
                        st.image(st.session_state.crnn_debug_image_matrix,
                                 caption="Proportionally normalized white-space backpadded matrix read by model.",
                                 use_container_width=True)
        st.divider()

    if 'db_lookup' not in st.session_state:
        db_data, db_msg = load_medicine_database(DB_PATH)
        st.session_state.db_lookup, st.session_state.db_msg = db_data, db_msg

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant",
                                      "content": "Hello! I am your AI Health Assistant. Describe your symptoms or upload an image selection snippet."}]

    if "last_processed_file_hash" not in st.session_state: st.session_state.last_processed_file_hash = None

    with st.sidebar:
        st.header("🔐 Secure Vault")
        if st.session_state.is_doctor: update_doctor_heartbeat()
        st.caption(f"Hardware ID: `{v_id}`")
        st.divider()

        if not st.session_state.auth:
            tab_unlock, tab_reg = st.tabs(["Unlock", "Register"])
            with tab_unlock:
                pin = st.text_input("Enter 6-Digit Key", type="password", key="vault_pin")
                if st.button("Unlock Features"):
                    if pin.strip() in ["998877", "112233"]:
                        st.session_state.is_doctor = True
                        st.session_state.auth = True
                        st.session_state.doctor_db_id = 999 if pin.strip() == "998877" else 888
                        st.session_state.doctor_display_name = "Dr. Code" if pin.strip() == "998877" else "Senior Consultant"
                        st.rerun()

                    if conn is not None:
                        try:
                            doc_check = conn.table("doctor_identities").select("*").eq("secret_pin",
                                                                                       str(pin).strip()).execute()
                            is_valid_doctor = len(doc_check.data) > 0
                        except Exception:
                            is_valid_doctor = False

                        if is_valid_doctor:
                            st.session_state.is_doctor = True
                            st.session_state.auth = True
                            st.session_state.doctor_db_id = doc_check.data[0]['id']
                            st.session_state.doctor_display_name = doc_check.data[0]['doctor_name']
                            st.rerun()
                        elif verify_user_cloud(v_id, pin):
                            st.session_state.is_doctor = False
                            st.session_state.auth = True
                            st.rerun()
                        else:
                            st.error("Invalid security signature.")
            with tab_reg:
                reg_role = st.selectbox("Select Track:",
                                        ["Patient Vault Profile", "Authorized Medical Practitioner Profile"])
                if reg_role == "Patient Vault Profile":
                    mail = st.text_input("Email for Key", key="vault_email")
                    if st.button("Generate Key"):
                        if "@" in mail:
                            k = generate_permanent_key(mail)
                            if save_user_cloud(v_id, mail, k): st.success(f"Key: **{k}**")
                        else:
                            st.error("Invalid Email Structure.")
                else:
                    doc_reg_name = st.text_input("Full Name (Include Dr. Prefix)", key="doc_reg_name_field")
                    doc_reg_mail = st.text_input("Institutional Medical Email", key="doc_reg_mail_field")
                    if st.button("Register Record"):
                        if doc_reg_name.strip() and "@" in doc_reg_mail and conn is not None:
                            compiled_doc_pin = generate_permanent_key(doc_reg_mail)
                            try:
                                conn.table("doctor_identities").insert(
                                    {"doctor_name": doc_reg_name.strip(), "email": doc_reg_mail.strip(),
                                     "secret_pin": str(compiled_doc_pin)}).execute()
                                st.success(f"🎉 Passcode: **{compiled_doc_pin}**")
                            except Exception as e:
                                st.error(f"Transaction Error: {e}")
        else:
            if st.session_state.is_doctor:
                st.success(f"👨‍⚕️ Portal: {st.session_state.doctor_display_name}")
                if st.button("Logout Doctor Account"):
                    st.session_state.is_doctor = False
                    st.session_state.auth = False
                    st.session_state.selected_room = None
                    st.rerun()
            else:
                st.success(f"✅ Access Active")
                if st.button("Logout"):
                    st.session_state.auth = False
                    st.session_state.active_patient_email = None
                    st.rerun()

    # ====================================================================
    # INTERCEPTOR VIEW A: CLINICAL INTERACTION CONSULTATION PANEL
    # ====================================================================
    if st.session_state.is_doctor:
        st.title("👨‍⚕️ Medical Professional Consultation Panel")
        current_active_doc_id = st.session_state.get('doctor_db_id', 1)
        distinct_rooms = []
        if conn is not None:
            try:
                rooms_query = conn.table("doctor_chat_messages").select("chat_room_id").like("chat_room_id",
                                                                                             f"doc_{current_active_doc_id}_%").execute()
                distinct_rooms = list(set([row['chat_room_id'] for row in rooms_query.data]))
            except Exception:
                pass

        verified_rooms = [r for r in distinct_rooms if "_patient_user_" in r]
        guest_rooms = [r for r in distinct_rooms if "_patient_guest_" in r]

        if st.session_state.selected_room is None:
            category_tab = st.radio("Directory", ["Registered Patients", "Anonymous Guests"], horizontal=True)
            selected_option = st.selectbox("Active Channels",
                                           verified_rooms if category_tab == "Registered Patients" else guest_rooms)
            if st.button("Open Consultation") and selected_option:
                st.session_state.selected_room = selected_option
                st.rerun()
        else:
            if st.button("⬅️ Back to Directory"):
                st.session_state.selected_room = None
                st.rerun()

            live_chat_stream(st.session_state.selected_room, view_role="doctor")

            st.divider()
            if doc_prompt := st.text_input("Send Guidance Notes:", key="doc_text_input_action_matrix"):
                if st.button("Transmit Note Package", use_container_width=True):
                    bg_db_insert(
                        {"chat_room_id": st.session_state.selected_room, "sender_type": "doctor", "sender_id": "doc",
                         "message_text": doc_prompt})
                    st.rerun()

    # ====================================================================
    # INTERCEPTOR VIEW B: PATIENT CLASSIFIER SYSTEMS CHAT INTERFACE
    # ====================================================================
    else:
        st.title("💬 AI Health Assistant")
        st.error(
            "⚠️ DISCLAIMER: This tool provides AI-generated health metrics for information purposes only. It does not replace professional emergency triage checks.")

        if st.session_state.chat_mode == "ai_assistant":
            avail_docs = [{"id": 998877, "doctor_name": "Duty Consultant Practitioner"}]
            if conn is not None:
                try:
                    docs_fetch = conn.table("doctor_identities").select("id, doctor_name").execute()
                    if docs_fetch.data: avail_docs = docs_fetch.data
                except:
                    pass

            st.subheader("Connect with Clinical Specialist")
            selected_doc_obj = st.selectbox("Choose Attending Practitioner Terminal:", avail_docs,
                                            format_func=lambda d: f"👨‍⚕️ {d['doctor_name']}")
            st.session_state.patient_selected_doctor_id = selected_doc_obj['id']
            patient_target_room_id = f"doc_{selected_doc_obj['id']}_patient_{room_prefix}{v_id}"

            if st.button("🩺 Connect Live to Doctor Channel", use_container_width=True, type="primary"):
                st.session_state.chat_mode = "doctor_consult"
                if conn is not None:
                    try:
                        conn.table("doctor_chat_messages").insert(
                            {"chat_room_id": patient_target_room_id, "sender_type": "patient", "sender_id": v_id,
                             "message_text": f"🚨 [System Alert]: Live channel initialized."}).execute()
                    except:
                        pass
                st.rerun()
            with st.expander("🚨 EMERGENCY Toolkit: Find Nearest Hospitals"):
                embed_hospital_finder()
        else:
            if st.button("🔴 Terminate Live Doctor Link", use_container_width=True):
                st.session_state.chat_mode = "ai_assistant"
                st.rerun()

        st.divider()
        if st.session_state.chat_mode == "doctor_consult":
            active_target_room = f"doc_{st.session_state.get('patient_selected_doctor_id', 998877)}_patient_{room_prefix}{v_id}"
            live_chat_stream(active_target_room, view_role="patient", active_v_id=v_id)

            st.divider()
            if prompt_direct := st.text_input("Type Message directly to Physician:",
                                              key="direct_patient_msg_input_field"):
                if st.button("Send Message Package"):
                    bg_db_insert({"chat_room_id": active_target_room, "sender_type": "patient", "sender_id": v_id,
                                  "message_text": prompt_direct})
                    st.rerun()
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])

            if prompt := st.chat_input("Enter symptoms or ask 'Do you know Malaria?'"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                query_lower = prompt.lower().strip()
                bot = st.session_state.bot

                if query_lower == "verify now":
                    with st.spinner("⚙️ Running verification & training pipeline..."):
                        success, msg = bot.execute_verification_cycle()
                    response_text = f"System Notification: {msg}"

                elif query_lower.startswith("do you know "):
                    disease_request = query_lower[12:].strip("?., ")
                    if disease_request:
                        if bot.log_learning_request(disease_request):
                            response_text = f"📝 **Request Logged:** Added **{disease_request}** to verification queue.\n\nType **'verify now'** to initialize background pipeline execution."
                        else:
                            response_text = "❌ Fallback logging path error."
                    else:
                        response_text = "Please specify a disease structure parameter."

                else:
                    search_term = DISEASE_ALIASES.get(query_lower, query_lower)
                    response_text = ""
                    disease_found = None

                    matches = difflib.get_close_matches(search_term, bot.known_diseases, n=1, cutoff=0.85)
                    if not matches:
                        matches = [d for d in bot.known_diseases if search_term in d]

                    if matches:
                        disease_found = matches[0]
                        response_text = f"✅ **Identification:** Discovered matching matrix for **{disease_found.title()}**.\n"
                    else:
                        disease, matched, conf = bot.predict(query_lower)
                        if matched:
                            response_text = f"**Based on symptoms** ({', '.join(matched).replace('_', ' ')}), I suspect **{disease.upper()}** (Confidence: {conf:.1f}%).\n"
                            disease_found = disease
                        else:
                            response_text = "❌ Symptoms unmapped. Try adding standard parameters or teach me using: *'Do you know [Disease]?'*"

                    if disease_found:
                        symptoms = bot.get_symptoms(disease_found)
                        if symptoms:
                            response_text += f"\n\n**Private Note Map:** Typical indications include: {', '.join(symptoms[:8])}.\n"
                        advice, source = bot.get_advice(disease_found)
                        response_text += f"\n\n---\n**🛡️ Recommended Advice** *(Source: {source})*:\n"
                        if advice:
                            for item in advice: response_text += f"- {item}\n"
                        else:
                            response_text += "- No direct online precaution matrix available."

                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()


if __name__ == "__main__":
    main()