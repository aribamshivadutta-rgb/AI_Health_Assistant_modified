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
import cv2
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pdf2image import convert_from_bytes
from rapidfuzz import process, fuzz

# Supabase dependency (Assuming you have this installed)
from st_supabase_connection import SupabaseConnection

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
# 1. FIXED PATH MATRIX & DATABASE UTILS
# ====================================================================
IS_ONLINE_DEPLOYMENT = os.path.exists("/mount/src") or not os.path.exists(r"C:\Users\Bubu")

if not IS_ONLINE_DEPLOYMENT:
    resolved_root = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System"
    MODEL_DIR = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System\models"
    DATA_DIR = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System\data\clean\chat_bot_clean"
    RAW_DIR = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System\data\raw"
    TEMP_DIR = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System\data\temp"
    PREPROCESS_SCRIPT = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System\scripts\chat_bot_preprocessing.py"
    TRAIN_SCRIPT = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System\scripts\train_lgbm.py"
else:
    possible_roots = [CURRENT_SCRIPT_DIR, os.path.dirname(CURRENT_SCRIPT_DIR),
                      os.path.dirname(os.path.dirname(CURRENT_SCRIPT_DIR))]
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


# --- AUTO-LOCATOR: Find the database anywhere in the project ---
def find_database_dynamically():
    for root_dir, _, files in os.walk(resolved_root):
        for file in files:
            if "Final_Compiled_Medicine_Database" in file:
                return os.path.join(root_dir, file)
    return None


DB_PATH = find_database_dynamically()

MEDICAL_DICTIONARY = ["Rx", "Tablet", "Capsule", "Amoxicillin", "Paracetamol", "Azithromycin", "Metformin", "Ibuprofen",
                      "Calbo-D", "Losita"]


def load_medicine_database(db_path):
    if db_path is None or not os.path.exists(db_path):
        return None, "File not found by Auto-Locator."
    try:
        try:
            df = pd.read_excel(db_path, engine='openpyxl')
        except Exception as e_xl:
            try:
                df = pd.read_csv(db_path, low_memory=False)
            except Exception as e_csv:
                return None, f"Excel Error: {e_xl} | CSV Error: {e_csv}"

        search_column = 'Medicine' if 'Medicine' in df.columns else df.columns[0]
        df['lookup_key'] = df[search_column].astype(str).str.strip().str.lower()
        df = df.drop_duplicates(subset=['lookup_key'], keep='first')
        return df.set_index('lookup_key').dropna(axis=1, how='all').to_dict(orient='index'), "Success"
    except Exception as e:
        return None, f"Processing Error: {str(e)}"


def fetch_medicine_details_fast(extracted_name, lookup_dict):
    if not lookup_dict: return None
    search_term = str(extracted_name).strip().lower()
    return lookup_dict.get(search_term, None)


# ====================================================================
# 2. CLOUD DATABASE MANAGEMENT (SUPABASE INTEGRATION)
# ====================================================================
try:
    conn = st.connection("supabase", type=SupabaseConnection, url="https://cwwoloupweulprxwibmp.supabase.co",
                         key="YOUR_SUPABASE_KEY_HERE")
except Exception:
    pass  # Fails safely if offline


def get_visitor_id():
    if 'visitor_id' not in st.session_state: st.session_state.visitor_id = str(uuid.getnode())
    return st.session_state.visitor_id


def verify_user_cloud(v_id, input_key):
    try:
        query = conn.table("user_identities").select("*").eq("visitor_id", v_id).eq("permanent_key",
                                                                                    str(input_key)).execute()
        return len(query.data) > 0
    except:
        return False


def generate_permanent_key(email):
    hash_obj = hashlib.sha256(email.strip().lower().encode())
    seed = int(hash_obj.hexdigest(), 16) % 10 ** 8
    random.seed(seed)
    return str(random.randint(100000, 999999))


def save_user_cloud(v_id, email, key):
    try:
        conn.table("user_identities").upsert({"visitor_id": v_id, "email": email, "permanent_key": str(key)}).execute()
        return True
    except:
        return False


# ====================================================================
# 3. ARCHITECTURE BLOCK (RESIDUAL CRNN)
# ====================================================================
class MedicalLabelEncoder:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
        self.chars = data['chars']
        self.lexicon = set(data['lexicon'])
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
        self.layer7 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.pool4 = nn.MaxPool2d((2, 1))
        self.hidden_size = 256
        self.num_layers = 2
        self.rnn = nn.LSTM(input_size=512 * 4, hidden_size=self.hidden_size, num_layers=self.num_layers,
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(self.hidden_size * 2, vocab_size)

    def forward(self, img_tensor, hx=None):
        x = self.pool4(self.layer7(self.spatial_drop2(self.pool3(self.layer6(self.layer5(self.spatial_drop1(
            self.pool2(self.layer4(self.layer3(self.pool1(self.layer2(self.layer1(img_tensor)))))))))))))
        b, c, h, w = x.size()
        rnn_out, _ = self.rnn(x.view(b, c * h, w).permute(0, 2, 1), hx)
        return self.fc(rnn_out).log_softmax(2)


# ====================================================================
# 4. OCR PIPELINE (UPDATED FOR HUMAN-IN-THE-LOOP)
# ====================================================================
class OCRReaderPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = None
        self.text_recognizer = None

        if os.path.exists(VOCAB_JSON):
            self.encoder = MedicalLabelEncoder(VOCAB_JSON)
            self.medical_dictionary = list(self.encoder.lexicon)
        else:
            self.medical_dictionary = MEDICAL_DICTIONARY
            self.encoder = None

        self.load_models()

    def load_models(self):
        if MedicalDetectorCNN is not None and os.path.exists(DETECTOR_WEIGHTS):
            self.detector = MedicalDetectorCNN(n_channels=1, n_classes=1).to(self.device)
            self.detector.load_state_dict(torch.load(DETECTOR_WEIGHTS, map_location=self.device))
            self.detector.eval()

        if self.encoder is not None and os.path.exists(CRNN_WEIGHTS):
            self.text_recognizer = MedicalResidualCRNN(self.encoder.vocab_size).to(self.device)
            raw_state_dict = torch.load(CRNN_WEIGHTS, map_location=self.device)
            sanitized_state_dict = {k.replace("module.", "").replace("model.", "").replace("text_recognizer.", ""): v
                                    for k, v in raw_state_dict.items()}
            self.text_recognizer.load_state_dict(sanitized_state_dict, strict=True)
            self.text_recognizer.eval()

    def detect_regions(self, image_input):
        """Step 1: U-Net proposes bounding boxes (Human-in-the-Loop Proposal Engine)"""
        raw_img = None
        if isinstance(image_input, str):
            raw_img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        else:
            filename = getattr(image_input, 'name', '').lower()
            if filename.endswith('.pdf'):
                pil_pages = convert_from_bytes(image_input.read())
                raw_img = cv2.cvtColor(np.array(pil_pages[0]), cv2.COLOR_RGB2GRAY)
            else:
                file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
                raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            image_input.seek(0)

        orig_h, orig_w = raw_img.shape[:2]
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        equalized_raw_img = clahe.apply(raw_img)
        resized_img = cv2.resize(equalized_raw_img, (512, 512))

        img_tensor = torch.from_numpy(resized_img / 255.0).unsqueeze(0).unsqueeze(0).float().to(self.device)
        mask = np.zeros((512, 512), dtype=np.uint8)

        if self.detector is not None:
            with torch.no_grad():
                mask_output = self.detector(img_tensor)
                mask = (mask_output.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8) * 255

        resized_mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        processed_mask = cv2.morphologyEx(resized_mask, cv2.MORPH_CLOSE,
                                          cv2.getStructuringElement(cv2.MORPH_RECT, (95, 8)))

        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        line_bounding_boxes = []

        # Grabs all text blocks for the user to review
        for ctr in sorted(contours, key=lambda c: cv2.boundingRect(c)[1]):
            xc, yc, wc, hc = cv2.boundingRect(ctr)
            if wc > 25 and hc > 10:
                line_bounding_boxes.append((xc, yc, wc, hc))

        return equalized_raw_img, line_bounding_boxes, cv2.resize(resized_mask, (512, 512))

    def _split_lines(self, block_crop):
        gray = cv2.cvtColor(block_crop, cv2.COLOR_BGR2GRAY) if len(block_crop.shape) == 3 else block_crop.copy()
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        horizontal_sum = np.sum(thresh, axis=1)
        lines, in_line, start_y = [], False, 0
        for idx, row_sum in enumerate(horizontal_sum):
            if not in_line and row_sum > 0:
                in_line, start_y = True, max(0, idx - 2)
            elif in_line and row_sum == 0:
                in_line, end_y = False, min(block_crop.shape[0], idx + 2)
                if (end_y - start_y) > 5: lines.append(block_crop[start_y:end_y, :])
        if in_line: lines.append(block_crop[start_y:, :])
        return lines if lines else [block_crop]

    def recognize_crop(self, block_crop):
        """Step 2: CRNN + Semantic Gatekeeper on user-confirmed crop"""
        tokenized_lines = self._split_lines(block_crop)
        final_text_lines, diagnostics = [], []

        for crop in tokenized_lines:
            target_w, target_h = 256, 64
            crnn_input = np.ones((target_h, target_w), dtype=np.uint8) * 255
            scale = target_h / float(crop.shape[0])
            nw, nh = int(crop.shape[1] * scale), target_h
            if nw > target_w:
                scale = target_w / float(crop.shape[1])
                nw, nh = target_w, int(crop.shape[0] * scale)

            resized_crop = cv2.resize(crop, (max(4, nw), max(4, nh)))
            start_y = max(0, (target_h - nh) // 2)
            crnn_input[start_y:start_y + nh, 5:5 + min(nw, target_w - 5)] = resized_crop[:, :min(nw, target_w - 5)]

            if np.mean(crnn_input) < 127: crnn_input = cv2.bitwise_not(crnn_input)
            crnn_tensor = torch.from_numpy((crnn_input.astype(np.float32) / 255.0 - 0.5) / 0.5).float().to(
                self.device).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                h0 = torch.zeros(4, 1, 256, dtype=torch.float32).to(self.device)
                logits = self.text_recognizer(crnn_tensor, (h0, h0))
                best_path = torch.argmax(logits.squeeze(0), dim=1).cpu().numpy()
                decoded_line = self.encoder.decode(best_path).strip()

                # =========================================================
                # 🎯 THE SEMANTIC GATEKEEPER
                # =========================================================
                text_lower = decoded_line.lower()
                pharma_keywords = ["tab", "cap", "syp", "inj", "rx", "mg", "ml", "calbo", "losita", "amox", "parac"]
                noise_keywords = ["dr.", "mbbs", "clinic", "fever", "headache", "bodyache", "date", "age", "sex", "c/o",
                                  "drink", "rest", "days", "food", "since"]

                is_medicine = False

                if any(kw in text_lower for kw in pharma_keywords):
                    is_medicine = True
                else:
                    match = process.extractOne(decoded_line, self.medical_dictionary, scorer=fuzz.WRatio)
                    if match and match[1] >= 80.0: is_medicine = True

                if any(noise in text_lower for noise in noise_keywords):
                    is_medicine = False

                if is_medicine and len(decoded_line) > 2 and "expected" not in text_lower:
                    final_text_lines.append(decoded_line)
                    diagnostics.append(decoded_line)

        return "\n".join(final_text_lines), diagnostics


# ====================================================================
# 5. CHATBOT AND CLASSIFICATION EXPERT LAYER
# ====================================================================
class MedicalAI:
    def __init__(self):
        self.model = None
        self.le = None
        self.known_symptoms = ["fever", "cough", "headache", "fatigue", "vomiting"]
        self.known_diseases = ["influenza", "common cold"]
        self.load_resources()

    def load_resources(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(LE_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.le = joblib.load(LE_PATH)
                if os.path.exists(FEAT_PATH): self.known_symptoms = pd.read_csv(FEAT_PATH, nrows=0).columns.tolist()
            except Exception:
                pass

    def predict(self, user_input):
        if self.model is None or self.le is None: return "Uncompiled Matrix", [], 0.0
        cleaned = re.sub(r'\b(and|or|I have|feeling|my|is)\b', '', user_input, flags=re.IGNORECASE)
        tokens = [s.strip().replace(" ", "_").lower() for s in cleaned.split(",")]
        if len(tokens) == 1 and " " in user_input: tokens = [s.strip().replace(" ", "_").lower() for s in
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
                        input_dict[k] = 1;
                        matched.append(k);
                        break

        if not matched: return None, [], 0
        pred_id = self.model.predict(pd.DataFrame([input_dict]))[0]
        conf = self.model.predict_proba(pd.DataFrame([input_dict]))[0][pred_id] * 100
        return self.le.inverse_transform([pred_id])[0], list(set(matched)), conf


# ====================================================================
# 6. STREAMLIT APPLICATION LOGIC (UPDATED WITH HITL UI)
# ====================================================================
def main():
    st.set_page_config(page_title="AI Health Assistant", layout="wide")

    if 'bot' not in st.session_state: st.session_state.bot = MedicalAI()
    if 'ocr_pipeline' not in st.session_state: st.session_state.ocr_pipeline = OCRReaderPipeline()
    if 'db_lookup' not in st.session_state:
        db_data, db_msg = load_medicine_database(DB_PATH)
        st.session_state.db_lookup = db_data
        st.session_state.db_msg = db_msg
    if "messages" not in st.session_state: st.session_state.messages = [{"role": "assistant",
                                                                         "content": "Hello! I can identify health risks and analyze medical documents. How can I help?"}]
    if 'auth' not in st.session_state: st.session_state.auth = False

    # HITL Session State Tracking
    if "last_file_hash" not in st.session_state: st.session_state.last_file_hash = None
    if "proposed_boxes" not in st.session_state: st.session_state.proposed_boxes = []
    if "full_image" not in st.session_state: st.session_state.full_image = None

    v_id = get_visitor_id()

    with st.sidebar:
        st.header("🔐 Secure Vault")
        st.caption(f"Hardware ID: `{v_id}`")

        st.divider()
        st.subheader("📡 Server Diagnostics")
        st.text(f"Is Online Host? {IS_ONLINE_DEPLOYMENT}")
        st.metric("Weights Found?", str(os.path.exists(DETECTOR_WEIGHTS)))
        st.metric("Database Loaded?", str(st.session_state.db_lookup is not None))

        if not st.session_state.db_lookup:
            st.error(f"DB Error: {st.session_state.get('db_msg', 'Unknown Error')}")

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
                        st.error("Invalid Key.")
            with tab_reg:
                mail = st.text_input("Email for Key", key="vault_email")
                if st.button("Generate Key"):
                    k = generate_permanent_key(mail)
                    if save_user_cloud(v_id, mail, k): st.success(f"Permanent Key: **{k}**")
        else:
            st.success("✅ Professional Access Active")
            if st.button("Logout"):
                st.session_state.auth = False
                st.rerun()

            st.divider()
            st.subheader("Clinical Data Upload")
            uploaded_file = st.file_uploader("Upload Patient Report", type=["pdf", "png", "jpg", "jpeg"])

            if uploaded_file is not None:
                raw_payload = uploaded_file.getvalue()
                file_hash = hashlib.md5(raw_payload).hexdigest()

                # STEP 1: U-Net Proposal Engine
                if st.session_state.last_file_hash != file_hash:
                    st.session_state.last_file_hash = file_hash
                    st.session_state.proposed_boxes = []

                    with st.spinner("Scanning document structure..."):
                        img, boxes, mask = st.session_state.ocr_pipeline.detect_regions(uploaded_file)
                        st.session_state.full_image = img
                        st.session_state.proposed_boxes = boxes
                        st.session_state.mask_preview = mask
                    st.success(f"Detected {len(boxes)} text blocks.")

            # STEP 2: The Human-in-the-Loop Validation UI
            if uploaded_file is not None and st.session_state.proposed_boxes:
                st.divider()
                st.subheader("🎯 Confirm Medication Area")
                st.caption(
                    "Review the isolated blocks below. Click 'Extract' on the block that contains the medicines.")

                # Scrolling container for the proposed crops
                with st.container(height=500):
                    for idx, (x, y, w, h) in enumerate(st.session_state.proposed_boxes):
                        pad = 8
                        crop_img = st.session_state.full_image[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]

                        if crop_img.size > 0:
                            st.image(crop_img, use_container_width=True)
                            if st.button(f"Extract Medicines", key=f"extract_{idx}", use_container_width=True):

                                with st.spinner("Analyzing text..."):
                                    ocr_text, diag = st.session_state.ocr_pipeline.recognize_crop(crop_img)

                                    if ocr_text.strip():
                                        st.session_state.messages.append(
                                            {"role": "user", "content": f"📋 *[Confirmed Extraction]:*\n{ocr_text}"})

                                        db_insights = ""
                                        for line in ocr_text.split("\n"):
                                            med_info = fetch_medicine_details_fast(line, st.session_state.db_lookup)
                                            if med_info:
                                                db_insights += f"💊 **{line.strip()}**\n* 👉 **Generic:** {med_info.get('Generic Name', 'N/A')}\n* 👉 **Purpose:** {med_info.get('Use/Purpose', 'N/A')}\n\n"

                                        response_text = "✅ Medicines extracted successfully via Semantic Gatekeeper."
                                        if db_insights: response_text += f"\n\n📚 **Database Matches:**\n\n{db_insights}"

                                        st.session_state.messages.append(
                                            {"role": "assistant", "content": response_text})
                                    else:
                                        st.warning(
                                            "Gatekeeper blocked this content. It appears to be symptoms or noise, not medicine.")

                                st.rerun()
                            st.divider()

    # Main Chat Interface
    st.title("💬 AI Health Assistant")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Enter symptoms..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        disease, matched, conf = st.session_state.bot.predict(prompt)
        if matched:
            st.session_state.messages.append({"role": "assistant",
                                              "content": f"**Suspected Diagnosis:** {disease.upper()} ({conf:.1f}%)\n\n**Symptoms:** {', '.join(matched).replace('_', ' ')}"})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "I couldn't recognize those symptoms."})
        st.rerun()


if __name__ == "__main__":
    main()