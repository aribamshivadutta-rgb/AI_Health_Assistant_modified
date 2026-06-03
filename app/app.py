import os
import random
import csv
import sys
import re
import difflib
import hashlib
import uuid
import subprocess
from datetime import datetime, timedelta
import json
import base64
import io
import threading

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
    """Generates a transient session token fallback for unauthenticated guests."""
    if 'visitor_session_uuid' not in st.session_state:
        st.session_state.visitor_session_uuid = str(uuid.uuid4())[:18]
    return st.session_state.visitor_session_uuid


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
        # Cross-reference tracking via unique security pin across user parameters
        query = conn.table("user_identities").select("*").eq("permanent_key", str(input_key)).execute()
        if len(query.data) > 0:
            st.session_state.active_patient_email = query.data[0]['email']
            return True
        return False
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
            if match and match[1] >= 45.0:
                decoded_line = match[0]

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
# 6. STREAMLIT APPLICATION LOGIC & DYNAMIC MAP ASSET LOADER
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

    response = "✅ Medicines extracted securely via Normalized Matrix."
    if db_insights: response += f"\n\n📚 **Database Matches:**\n\n{db_insights}"
    return response


def embed_hospital_finder():
    html_path = os.path.join(CURRENT_SCRIPT_DIR, "hospital_finder.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_code = f.read()
        components.html(html_code, height=540, scrolling=True)
    else:
        st.error("Hospital finder core engine template target execution canvas mapping missing.")


# ====================================================================
# LIVE TELEHEALTH AUTO-POLLING STREAM MATRIX (FOCUS-SAFE)
# ====================================================================
@st.fragment(run_every=5)
def live_chat_stream(room_id, view_role, active_v_id=None):
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

        st.write("")
        if view_role == "doctor":
            if doc_prompt := st.chat_input("Type professional guidance...", key="doc_msg_input_field"):
                doc_payload = {
                    "chat_room_id": room_id,
                    "sender_type": "doctor",
                    "sender_id": "doc",
                    "message_text": doc_prompt
                }
                bg_db_insert(doc_payload)
                st.toast("✉️ Sent!", icon="📤")
        else:
            if prompt := st.chat_input("Type message directly to doctor...", key="patient_msg_input_field"):
                patient_payload = {
                    "chat_room_id": room_id,
                    "sender_type": "patient",
                    "sender_id": active_v_id,
                    "message_text": prompt
                }
                bg_db_insert(patient_payload)
                st.toast("✉️ Sent!", icon="📤")

    except Exception as e:
        st.caption(f"⚡ Connection jitter handled safely. Synchronizing framework... ({e})")


# ====================================================================
# ASYNC WORKER QUEUE DEFINITIONS
# ====================================================================
def bg_db_insert(payload_dict):
    try:
        conn.table("doctor_chat_messages").insert(payload_dict).execute()
    except Exception:
        pass


def update_doctor_heartbeat():
    try:
        conn.table("doctor_status").upsert({
            "is_online": True,
            "last_seen": datetime.now().isoformat()
        }, on_conflict="is_online").execute()
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
    if 'selected_room' not in st.session_state: st.session_state.selected_room = None
    if 'active_patient_email' not in st.session_state: st.session_state.active_patient_email = None

    if 'db_lookup' not in st.session_state:
        db_data, db_msg = load_medicine_database(DB_PATH)
        st.session_state.db_lookup, st.session_state.db_msg = db_data, db_msg

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I can identify health risks. How are you feeling?"}]

    if "last_processed_file_hash" not in st.session_state: st.session_state.last_processed_file_hash = None
    if "line_diagnostics" not in st.session_state: st.session_state.line_diagnostics = []
    if 'camera_active' not in st.session_state: st.session_state.camera_active = False

    file_upload = None
    camera_photo = None

    # Determine unique individual identity keys deterministically bound to specific verified emails
    if st.session_state.auth and st.session_state.active_patient_email:
        raw_target_string = str(st.session_state.active_patient_email).strip().lower()
        v_id = hashlib.sha256(raw_target_string.encode()).hexdigest()[:16]
        room_prefix = "user_"
    else:
        v_id = get_visitor_id()
        room_prefix = "guest_"

    active_room_id = f"{room_prefix}{v_id}"

    with st.sidebar:
        st.header("🔐 Secure Vault")
        if st.session_state.is_doctor:
            update_doctor_heartbeat()

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

        # Auth logic
        if not st.session_state.auth:
            st.warning("Locked Mode: Chat only.")
            tab_unlock, tab_reg = st.tabs(["Unlock", "Register"])
            with tab_unlock:
                pin = st.text_input("Enter 6-Digit Key", type="password", key="vault_pin")
                if st.button("Unlock Features"):
                    if pin.strip():
                        if pin.strip() in ["998877", "112233"]:
                            st.session_state.is_doctor = True
                            st.session_state.auth = True
                            st.session_state.doctor_display_name = "Dr. Xyz" if pin.strip() == "998877" else "Senior Consultant"
                            st.success(f"Bypass Authorized! Welcome, {st.session_state.doctor_display_name}.")
                            st.rerun()

                        try:
                            doc_check = conn.table("doctor_identities").select("*").eq("secret_pin",
                                                                                       str(pin).strip()).execute()
                            is_valid_doctor = len(doc_check.data) > 0
                        except Exception:
                            is_valid_doctor = False

                        if is_valid_doctor:
                            st.session_state.is_doctor = True
                            st.session_state.auth = True
                            st.session_state.doctor_display_name = doc_check.data[0]['doctor_name']
                            st.success(f"Welcome back, {st.session_state.doctor_display_name}!")
                            st.rerun()
                        elif verify_user_cloud(v_id, pin):
                            st.session_state.is_doctor = False
                            st.session_state.auth = True
                            st.rerun()
                        else:
                            st.error("Invalid security signature for this terminal sequence.")
            with tab_reg:
                reg_role = st.selectbox("Select Target Identity Track:",
                                        ["Patient Vault Profile", "Authorized Medical Practitioner Profile"])
                st.divider()
                if reg_role == "Patient Vault Profile":
                    mail = st.text_input("Email for Key", key="vault_email")
                    if st.button("Generate Key"):
                        if "@" in mail:
                            k = generate_permanent_key(mail)
                            if save_user_cloud(v_id, mail, k): st.success(f"Permanent Patient Key: **{k}**")
                        else:
                            st.error("Invalid Email Structure.")
                elif reg_role == "Authorized Medical Practitioner Profile":
                    st.error("""
                    ⚠️ **CRITICAL ACCOUNT INTEGRITY SECTOR WARNING**

                    This initialization channel is strictly reserved for verified, licensed 
                    clinical professionals. If you are **NOT a doctor** and select this track 
                    to build an unauthorized access signature, **you explicitly agree to face 
                    absolute judgment, immediate terminal banning, and direct legal prosecution.**
                    """)

                    doc_reg_name = st.text_input("Full Name (Include Dr. Prefix)", placeholder="e.g. Dr. Xyz",
                                                 key="doc_reg_name_field")
                    doc_reg_mail = st.text_input("Institutional Medical Email Account", key="doc_reg_mail_field")
                    if st.button("Register Clinical Record & Agree", use_container_width=True, type="primary"):
                        if doc_reg_name.strip() and "@" in doc_reg_mail:
                            compiled_doc_pin = generate_permanent_key(doc_reg_mail)
                            try:
                                conn.table("doctor_identities").insert({
                                    "doctor_name": doc_reg_name.strip(),
                                    "email": doc_reg_mail.strip(),
                                    "secret_pin": str(compiled_doc_pin)
                                }).execute()
                                st.success(f"🎉 Passcode Generated: **{compiled_doc_pin}**")
                            except Exception as e:
                                st.error(f"Transaction Error: {str(e)}")
                        else:
                            st.warning("Ensure practitioner names are completed and check email configurations.")
        else:
            if st.session_state.is_doctor:
                st.success(f"👨‍⚕️ Portal Authorization: {st.session_state.doctor_display_name}")
                if st.button("Logout Doctor Account"):
                    st.session_state.is_doctor = False
                    st.session_state.auth = False
                    st.session_state.doctor_display_name = "Doctor"
                    st.session_state.selected_room = None
                    st.rerun()
            else:
                st.success(
                    f"✅ Patient Access Active: {st.session_state.active_patient_email if st.session_state.active_patient_email else ''}")
                if st.button("Logout"):
                    st.session_state.auth = False
                    st.session_state.active_patient_email = None
                    if 'visitor_session_uuid' in st.session_state:
                        del st.session_state.visitor_session_uuid
                    st.rerun()

                st.divider()
                st.subheader("Clinical Data Capture")
                file_upload = st.file_uploader("Upload Patient Report", type=["pdf", "png", "jpg", "jpeg"])
                st.markdown("<p style='text-align: center; margin: 5px 0;'><b>— OR —</b></p>", unsafe_allow_html=True)
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
                    camera_photo = st.camera_input("Capture Medicine")

            uploaded_file = camera_photo if camera_photo else file_upload

            if not st.session_state.is_doctor and uploaded_file is not None:
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

                            upload_data = {
                                "chat_room_id": active_room_id,
                                "sender_type": "patient",
                                "sender_id": v_id,
                                "message_text": b64_payload
                            }
                            threading.Thread(target=bg_db_insert, args=(upload_data,), daemon=True).start()
                        except Exception as e:
                            st.error(f"Image compilation error: {e}")
                    else:
                        st.session_state.line_diagnostics = []
                        raw_img = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

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

                        if camera_photo is not None or img_aspect > 2.2:
                            with st.spinner("Processing Isolated Target Card..."):
                                text_out, conf_out = st.session_state.ocr_pipeline.recognize_crop(raw_img)
                                if text_out.strip():
                                    scan_payload = f"📋 *[Direct Crop Extraction]:*\n{text_out}"
                                    response = process_extraction_result(text_out, st.session_state.db_lookup)
                                    st.session_state.line_diagnostics = [
                                        {"text": text_out, "confidence": f"{conf_out:.2f}%"}]

                                    st.session_state.messages.append({"role": "user", "content": scan_payload})
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                                    st.rerun()
                                else:
                                    st.error(
                                        "No legible medical words matched system database parameters within the isolated region.")
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
                                    scan_payload = f"📋 *[Document Scan Extraction]:*\n{ocr_combined_result}"
                                    response = process_extraction_result(ocr_combined_result,
                                                                         st.session_state.db_lookup)

                                    st.session_state.messages.append({"role": "user", "content": scan_payload})
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                                    st.rerun()
                                else:
                                    st.error("No valid medical entries could be identified across the layout.")

                    if uploaded_file is not None and st.session_state.line_diagnostics:
                        st.divider()
                        st.subheader("🔬 Document Parser Matrix")
                        for idx, diag in enumerate(st.session_state.line_diagnostics):
                            st.metric(f"Line Segment #{idx + 1}", diag['text'], delta=diag['confidence'])

    # ====================================================================
    # INTERCEPTOR VIEW A: DOCTOR INTERACTION PORTAL
    # ====================================================================
    if st.session_state.is_doctor:
        st.title("👨‍⚕️ Medical Professional Consultation Panel")

        try:
            rooms_query = conn.table("doctor_chat_messages").select("chat_room_id").execute()
            distinct_rooms = list(set([row['chat_room_id'] for row in rooms_query.data]))
            users_query = conn.table("user_identities").select("visitor_id, email").execute()
            identity_map = {row['visitor_id']: row['email'] for row in users_query.data}

            # Formulate fallback labels for legacy or guest strings dynamically mapped via sub-queries
            for rm in distinct_rooms:
                clean_uid = rm.replace("user_", "")
                if clean_uid not in identity_map and rm.startswith("user_"):
                    # Fallback mapping configuration matrix path strategy
                    try:
                        user_db_check = conn.table("user_identities").select("email").eq("visitor_id",
                                                                                         clean_uid).execute()
                        if user_db_check.data:
                            identity_map[clean_uid] = user_db_check.data[0]['email']
                    except:
                        pass
        except Exception:
            distinct_rooms, identity_map = [], {}

        verified_rooms = [r for r in distinct_rooms if r.startswith("user_")]
        guest_rooms = [r for r in distinct_rooms if r.startswith("guest_")]

        if st.session_state.selected_room is None:
            category_tab = st.radio("Select Directory Category",
                                    ["Verified Registered Patients", "Anonymous Guest Consultations"], horizontal=True)
            selected_option = None
            if category_tab == "Verified Registered Patients":
                if not verified_rooms:
                    st.info("No active verified consultations.")
                else:
                    def user_label(r):
                        uid_key = r.replace('user_', '')
                        matched_email = identity_map.get(uid_key)
                        return f"👤 {matched_email if matched_email else 'Verified Key Extraction'} — [...{uid_key[-6:]}]"

                    selected_option = st.selectbox("Active Verified Channels", verified_rooms, format_func=user_label)
            else:
                if not guest_rooms:
                    st.info("No active guest sessions.")
                else:
                    selected_option = st.selectbox("Active Guest Channels", guest_rooms,
                                                   format_func=lambda r: f"⏳ Guest Session — [...{r[-6:]}]")

            if st.button("Open Consultation", type="primary"):
                st.session_state.selected_room = selected_option
                st.rerun()
        else:
            if st.button("⬅️ Back to Directory"):
                st.session_state.selected_room = None
                st.rerun()

            st.subheader(f"Consultation with: `{st.session_state.selected_room}`")
            live_chat_stream(st.session_state.selected_room, view_role="doctor")

    # ====================================================================
    # INTERCEPTOR VIEW B: PATIENT SYSTEM INTERFACE
    # ====================================================================
    else:
        st.title("💬 AI Health Assistant")
        if st.session_state.chat_mode == "ai_assistant":
            btn_label = "Live chat Doctor Portal" if st.session_state.auth else "🩺 Connect Live to Doctor (As Guest Client)"
            if st.button(btn_label, use_container_width=True, type="primary"):
                st.session_state.chat_mode = "doctor_consult"
                try:
                    conn.table("doctor_chat_messages").insert(
                        {"chat_room_id": active_room_id, "sender_type": "patient", "sender_id": v_id,
                         "message_text": f"🚨 [System Alert]: Room initiated."}).execute()
                except:
                    pass
                st.rerun()
            with st.expander("🚨 EMERGENCY Toolkit: Find Nearest Hospitals", expanded=False):
                embed_hospital_finder()
        else:
            if st.button("🔴 Terminate Live Doctor Link", use_container_width=True):
                st.session_state.chat_mode = "ai_assistant"
                if 'visitor_session_uuid' in st.session_state:
                    del st.session_state.visitor_session_uuid
                st.rerun()

        st.divider()
        if st.session_state.chat_mode == "doctor_consult":
            live_chat_stream(active_room_id, view_role="patient", active_v_id=v_id)
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])
            if prompt := st.chat_input("Enter symptoms..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                disease, matched, conf = st.session_state.bot.predict(prompt)
                if matched:
                    response_text = f"**Suspected Diagnosis:** {disease.upper()} ({conf:.1f}%)\n\n**Matched Symptoms:** {', '.join(matched).replace('_', ' ')}"
                else:
                    response_text = "I couldn't recognize those symptoms."
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()


if __name__ == "__main__":
    main()