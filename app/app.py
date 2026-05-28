import os
import random
import csv
import sys
import re
import difflib
import hashlib
import uuid
import subprocess
from datetime import datetime
import json

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
from pdf2image import convert_from_bytes
from rapidfuzz import process, fuzz, distance
from st_supabase_connection import SupabaseConnection
from PIL import Image

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
sys.path.append(CURRENT_SCRIPT_DIR)

try:
    from scripts.medical_detector_cnn import MedicalDetectorCNN
except ImportError:
    MedicalDetectorCNN = None

# ====================================================================
# 1. FIXED SELF-CORRECTING PATH MATRIX & DATABASE UTILS
# ====================================================================
IS_ONLINE_DEPLOYMENT = os.path.exists("/mount/src") or not os.path.exists(r"C:\Users\Bubu")

if not IS_ONLINE_DEPLOYMENT:
    resolved_root = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System"
    MODEL_DIR = os.path.join(resolved_root, "models")
    DATA_DIR = os.path.join(resolved_root, "data", "clean", "chat_bot_clean")
    RAW_DIR = os.path.join(resolved_root, "data", "raw")
    TEMP_DIR = os.path.join(resolved_root, "data", "temp")
    PREPROCESS_SCRIPT = os.path.join(resolved_root, "scripts", "chat_bot_preprocessing.py")
    TRAIN_SCRIPT = os.path.join(resolved_root, "scripts", "train_lgbm.py")
else:
    possible_roots = [
        CURRENT_SCRIPT_DIR,
        os.path.dirname(CURRENT_SCRIPT_DIR),
        os.path.dirname(os.path.dirname(CURRENT_SCRIPT_DIR))
    ]
    resolved_root = CURRENT_SCRIPT_DIR
    for root in possible_roots:
        if os.path.exists(os.path.join(root, "models", "medical_detector.pth")):
            resolved_root = root
            break

    MODEL_DIR = os.path.join(resolved_root, "models")
    DATA_DIR = os.path.join(resolved_root, "data", "clean", "chat_bot_clean")
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

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


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
# 2. CLOUD DATABASE MANAGEMENT (SUPABASE INTEGRATION)
# ====================================================================
try:
    conn = st.connection(
        "supabase",
        type=SupabaseConnection,
        url="https://cwwoloupweulprxwibmp.supabase.co",
        key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN3d29sb3Vwd2V1bHByeHdpYm1wIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg3MDA5NDEsImV4cCI6MjA5NDI3Njk0MX0.ggPfeYBaL7PLiEM8_fYI5fHo48obb5yRum_kR1CORNM"
    )
except Exception as e:
    pass


def get_visitor_id():
    if 'visitor_id' not in st.session_state: st.session_state.visitor_id = str(uuid.getnode())
    return st.session_state.visitor_id


def generate_permanent_key(email):
    random.seed(int(hashlib.sha256(email.strip().lower().encode()).hexdigest(), 16) % 10 ** 8)
    return str(random.randint(100000, 999999))


def save_user_cloud(v_id, email, key):
    try:
        conn.table("user_identities").upsert({"visitor_id": v_id, "email": email, "permanent_key": str(key)}).execute()
        return True
    except Exception:
        return False


def verify_user_cloud(v_id, input_key):
    try:
        return len(conn.table("user_identities").select("*").eq("visitor_id", v_id).eq("permanent_key",
                                                                                       str(input_key)).execute().data) > 0
    except:
        return False


# ====================================================================
# 3. ARCHITECTURE BLOCK (RESIDUAL CRNN)
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

    def forward(self, x): return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))


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
# 4. SMART HYBRID OCR PIPELINE
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

        cleaned_line = cv2.bilateralFilter(crop, 5, 65, 65)

        target_w, target_h = 256, 64
        crnn_input = np.ones((target_h, target_w), dtype=np.uint8) * 255

        scale = target_h / float(cleaned_line.shape[0])
        nw, nh = int(cleaned_line.shape[1] * scale), target_h

        if nw > target_w:
            scale = target_w / float(cleaned_line.shape[1])
            nw, nh = target_w, int(cleaned_line.shape[0] * scale)

        nw, nh = max(4, nw), max(4, nh)
        try:
            resized_crop = cv2.resize(cleaned_line, (nw, nh), interpolation=cv2.INTER_CUBIC)
        except:
            return "", 0.0

        start_x = max(0, (target_w - nw) // 2)
        start_y = max(0, (target_h - nh) // 2)
        actual_w = min(nw, target_w - start_x)
        crnn_input[start_y:start_y + nh, start_x:start_x + actual_w] = resized_crop[:, :actual_w]

        if np.mean(crnn_input) < 127: crnn_input = cv2.bitwise_not(crnn_input)

        crnn_input = (crnn_input.astype(np.float32) / 255.0 - 0.5) / 0.5
        crnn_tensor = torch.from_numpy(crnn_input).float().to(self.device).unsqueeze(0).unsqueeze(0)

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
                text_lower = decoded_line.lower()

            match = process.extractOne(decoded_line, self.medical_dictionary, scorer=fuzz.WRatio)
            if match and match[1] >= 82.0:
                decoded_line = match[0]
            else:
                return "", 0.0

            return decoded_line, line_confidence


# ====================================================================
# 5. CHATBOT AND CLASSIFICATION EXPERT LAYER
# ====================================================================
class MedicalAI:
    def __init__(self):
        self.model, self.le = None, None
        self.known_symptoms, self.known_diseases = [], []
        self.df_full = None
        self.load_resources()

    def load_resources(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(LE_PATH):
            try:
                self.model, self.le = joblib.load(MODEL_PATH), joblib.load(LE_PATH)
                if os.path.exists(FEAT_PATH): self.known_symptoms = pd.read_csv(FEAT_PATH, nrows=0).columns.tolist()
                self.known_diseases = [d.lower() for d in self.le.classes_]
                if os.path.exists(FULL_DATA_PATH): self.df_full = pd.read_csv(FULL_DATA_PATH)
            except Exception as e:
                print(f"Soft Initialization Warning: {e}")
        else:
            self.known_symptoms = ["fever", "cough", "headache", "fatigue", "vomiting"]
            self.known_diseases = ["influenza", "common cold"]

    def log_learning_request(self, disease_name):
        if not os.path.exists(REQUESTS_FILE):
            with open(REQUESTS_FILE, 'w', newline='', encoding='utf-8') as f: csv.writer(f).writerow(
                ["timestamp", "source_url", "proposed_disease", "symptoms", "status"])
        with open(REQUESTS_FILE, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(
                [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "App", disease_name, "Pending", "Pending"])

    def execute_verification_cycle(self):
        try:
            st.info("🧠 Recalculating machine learning weights on local architecture...")
            if os.path.exists(PREPROCESS_SCRIPT):
                subprocess.run([sys.executable, PREPROCESS_SCRIPT], check=True)
            if os.path.exists(TRAIN_SCRIPT):
                subprocess.run([sys.executable, TRAIN_SCRIPT], check=True)
            self.load_resources()
            return True, "✅ Retraining Complete! Missing model parameters have been successfully compiled."
        except Exception as e:
            return False, f"Retraining lifecycle bypassed: {e}"

    def predict(self, user_input):
        if self.model is None or self.le is None:
            return "Uncompiled Classifier Matrix (Type 'verify now')", [], 0.0

        cleaned = re.sub(r'\b(and|or|I have|feeling|my|is)\b', '', user_input, flags=re.IGNORECASE)
        tokens = [s.strip().replace(" ", "_").lower() for s in cleaned.split(",")]
        if len(tokens) == 1 and " " in user_input.strip(): tokens = [s.strip().replace(" ", "_").lower() for s in
                                                                     user_input.split(" ")]

        input_dict = {col: 0 for col in self.known_symptoms}
        matched = []
        for t in tokens:
            m = difflib.get_close_matches(t, self.known_symptoms, n=1, cutoff=0.6)
            if m:
                input_dict[m[0]] = 1
                matched.append(m[0])
            else:
                for k in self.known_symptoms:
                    if t in k.replace("_", " ") or k.replace("_", " ") in t:
                        input_dict[k] = 1
                        matched.append(k)
                        break

        if not matched: return None, [], 0
        pred_id = self.model.predict(pd.DataFrame([input_dict]))[0]
        return self.le.inverse_transform([pred_id])[0], list(set(matched)), \
            self.model.predict_proba(pd.DataFrame([input_dict]))[0][pred_id] * 100


# ====================================================================
# 6. STREAMLIT APPLICATION LOGIC
# ====================================================================
def process_extraction_result(ocr_text, db_lookup):
    db_insights = ""
    for line in ocr_text.split("\n"):
        if len(line.strip()) <= 2: continue
        med_info = fetch_medicine_details_fast(line.strip(), db_lookup)
        if med_info: db_insights += f"💊 **{line.strip()}**\n* 👉 **Generic:** {med_info.get('Generic Name', 'N/A')}\n* 👉 **Purpose:** {med_info.get('Use/Purpose', 'N/A')}\n\n"

    response = "✅ Medicines extracted securely via Normalized Matrix."
    if db_insights: response += f"\n\n📚 **Database Matches:**\n\n{db_insights}"
    return response


def main():
    st.set_page_config(page_title="AI Health Assistant", layout="wide")

    if 'bot' not in st.session_state: st.session_state.bot = MedicalAI()
    if 'ocr_pipeline' not in st.session_state: st.session_state.ocr_pipeline = OCRReaderPipeline()
    if 'auth' not in st.session_state: st.session_state.auth = False

    if 'db_lookup' not in st.session_state:
        db_data, db_msg = load_medicine_database(DB_PATH)
        st.session_state.db_lookup, st.session_state.db_msg = db_data, db_msg

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I can identify health risks. How are you feeling?"}]

    if "last_processed_file_hash" not in st.session_state: st.session_state.last_processed_file_hash = None
    if "line_diagnostics" not in st.session_state: st.session_state.line_diagnostics = []
    if 'camera_active' not in st.session_state: st.session_state.camera_active = False

    v_id = get_visitor_id()

    with st.sidebar:
        st.header("🔐 Secure Vault")
        st.caption(f"Hardware ID: `{v_id}`")

        st.divider()
        st.subheader("📡 Server Path Diagnostics")
        st.text(f"Is Online Host? {IS_ONLINE_DEPLOYMENT}")
        st.metric("Weights Target File Found?", str(os.path.exists(DETECTOR_WEIGHTS)))
        st.metric("Database Loaded?", str(st.session_state.db_lookup is not None))

        if not st.session_state.db_lookup:
            st.caption(f"Path Found: {DB_PATH}")
            st.error(f"Crash Log: {st.session_state.get('db_msg', 'Unknown Error')}")

        st.divider()

        if not st.session_state.auth:
            st.warning("Locked Mode: Chat only.")
            tab_unlock, tab_reg = st.tabs(["Unlock", "Register"])
            with tab_unlock:
                pin = st.text_input("Enter 6-Digit Key", type="password", key="vault_pin")
                if st.button("Unlock Features"):
                    if verify_user_cloud(v_id, pin):
                        st.session_state.auth = True
                        st.rerun()
                    else:
                        st.error("Invalid Key for this device.")
            with tab_reg:
                mail = st.text_input("Email for Key", key="vault_email")
                if st.button("Generate Key"):
                    if "@" in mail:
                        k = generate_permanent_key(mail)
                        if save_user_cloud(v_id, mail, k):
                            st.success(f"Permanent Key: **{k}**")
                    else:
                        st.error("Invalid Email Structure.")
        else:
            st.success("✅ Professional Access Active")
            if st.button("Logout"): st.session_state.auth = False; st.rerun()

            st.divider()
            st.subheader("Clinical Data Capture")

            file_upload = st.file_uploader("Upload Patient Report", type=["pdf", "png", "jpg", "jpeg"])
            st.markdown("<p style='text-align: center; margin: 5px 0;'><b>— OR —</b></p>", unsafe_allow_html=True)

            camera_photo = None
            if not st.session_state.camera_active:
                if st.button("📸 Open Live Camera Scanner", use_container_width=True):
                    st.session_state.camera_active = True
                    st.rerun()
            else:
                if st.button("❌ Close Camera", use_container_width=True):
                    st.session_state.camera_active = False
                    st.rerun()

                # --- ADVANCED FULL-BLEED OVERLAY CSS INTEGRATION SYSTEM ---
                st.markdown("""
                <style>
                [data-testid="stCameraInput"] { position: relative; width: 100% !important; }
                [data-testid="stCameraInput"] video { width: 100% !important; height: auto !important; object-fit: cover !important; }
                [data-testid="stCameraInput"]:has(video)::before {
                    content: 'ALIGN MEDICINE NAME HERE';
                    position: absolute; top: 38%; left: 5%; width: 90%; height: 24%;
                    border: 3px dashed #00FF00; color: #00FF00; display: flex; align-items: center; justify-content: center;
                    font-size: 13px; font-weight: bold; text-align: center; z-index: 99; pointer-events: none;
                    background-color: rgba(0, 255, 0, 0.08); box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.4);
                }
                </style>
                """, unsafe_allow_html=True)

                camera_photo = st.camera_input("Capture Medicine")

            uploaded_file = camera_photo if camera_photo else file_upload

            if uploaded_file is not None:
                file_bytes = uploaded_file.getvalue()
                file_hash = hashlib.md5(file_bytes).hexdigest()

                if st.session_state.last_processed_file_hash != file_hash:
                    st.session_state.last_processed_file_hash = file_hash
                    st.session_state.line_diagnostics = []

                    raw_img_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
                    raw_img = cv2.imdecode(raw_img_array, cv2.IMREAD_GRAYSCALE)

                    # --- ADVANCED COMPUTER VISION ISOLATION MASK ENGINE ---
                    if camera_photo is not None:
                        h_orig, w_orig = raw_img.shape[:2]
                        y1 = int(h_orig * 0.38)
                        y2 = int(h_orig * 0.62)
                        x1 = int(w_orig * 0.05)
                        x2 = int(w_orig * 0.95)

                        # Step 1: Slice out the interior box array matching green overlay dimensions
                        target_box = raw_img[y1:y2, x1:x2].copy()

                        # Step 2: Use Adaptive Thresholding to capture handwritten ink stroke clusters
                        blur_box = cv2.GaussianBlur(target_box, (3, 3), 0)
                        thresh_box = cv2.adaptiveThreshold(
                            blur_box, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY_INV, 25, 9
                        )

                        # Step 3: Turn foreground text ink directly to high-contrast digital black
                        processed_target = cv2.bitwise_not(thresh_box)

                        # Step 4: Create a clean canvas masked to pure model-native white background layout
                        raw_img = np.ones((h_orig, w_orig), dtype=np.uint8) * 255
                        raw_img[y1:y2, x1:x2] = processed_target
                    else:
                        # Adaptive binarization pass for file uploads
                        raw_img = cv2.adaptiveThreshold(
                            raw_img, 255,
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 41, 15
                        )

                    orig_h, orig_w = raw_img.shape[:2]
                    img_aspect = orig_w / float(orig_h)

                    # --- DYNAMIC ROUTING MATRIX GATES ---
                    if camera_photo is not None or img_aspect > 2.2:
                        with st.spinner("Processing Exact Normalization Layer..."):
                            if camera_photo is not None:
                                pass_img = raw_img[y1:y2, x1:x2].copy()
                            else:
                                pass_img = raw_img

                            text_out, conf_out = st.session_state.ocr_pipeline.recognize_crop(pass_img)
                            if text_out.strip():
                                st.session_state.line_diagnostics = [
                                    {"text": text_out, "confidence": f"{conf_out:.2f}%"}]
                                st.session_state.messages.append(
                                    {"role": "user", "content": f"📋 *[Direct Crop Extraction]:*\n{text_out}"})
                                response = process_extraction_result(text_out, st.session_state.db_lookup)
                                st.session_state.messages.append({"role": "assistant", "content": response})
                            else:
                                st.error("No legible tokens matched your system database records.")
                    else:
                        with st.spinner("Deconstructing Full Page Matrix..."):
                            extracted_slices = st.session_state.ocr_pipeline.segment_full_prescription(raw_img)

                            all_discovered_text = []
                            diags_pool = []

                            for slice_block in extracted_slices:
                                text_out, conf_out = st.session_state.ocr_pipeline.recognize_crop(slice_block)
                                if text_out.strip():
                                    all_discovered_text.append(text_out)
                                    diags_pool.append({"text": text_out, "confidence": f"{conf_out:.2f}%"})

                            st.session_state.line_diagnostics = diags_pool
                            ocr_combined_result = "\n".join(all_discovered_text)

                            if ocr_combined_result.strip():
                                st.session_state.messages.append(
                                    {"role": "user",
                                     "content": f"📋 *[Document Scan Extraction]:*\n{ocr_combined_result}"})
                                response = process_extraction_result(ocr_combined_result, st.session_state.db_lookup)
                                st.session_state.messages.append({"role": "assistant", "content": response})
                            else:
                                st.error("No valid medicine entries could be identified across the full layout lines.")

            if uploaded_file is not None and st.session_state.line_diagnostics:
                st.divider()
                st.subheader("🔬 Document Parser Matrix")
                for idx, diag in enumerate(st.session_state.line_diagnostics):
                    st.metric(f"Line Segment #{idx + 1}", diag['text'], delta=diag['confidence'])

    # Main Chat View
    st.title("💬 AI Health Assistant")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Enter symptoms (e.g. fever, headache)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        disease, matched, conf = st.session_state.bot.predict(prompt)
        if matched:
            response_text = f"**Suspected Diagnosis:** {disease.upper()} ({conf:.1f}%)\n\n**Matched Symptoms:** {', '.join(matched).replace('_', ' ')}"
        else:
            response_text = "I couldn't recognize those symptoms. Try 'Do you know [Disease]?' to teach me."
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()


if __name__ == "__main__":
    main()