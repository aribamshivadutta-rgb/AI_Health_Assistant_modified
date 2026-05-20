import os
import sys
import subprocess
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import csv
import time

# =======================
# CONFIGURATION
# =======================
# 1. Dynamic Path Detection
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 2. Updated Data Paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "clean", "chat_bot_clean")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
TEMP_DIR = os.path.join(PROJECT_ROOT, "data", "temp")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

REQUESTS_FILE = os.path.join(TEMP_DIR, "unverified_diseases.csv")
LEARNED_DATA_FILE = os.path.join(RAW_DIR, "learned_user_data.csv")
ADVICE_DB_PATH = os.path.join(RAW_DIR, "who_data_clean.csv")
FEAT_PATH = os.path.join(DATA_DIR, "X_preprocessed.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "lgbm_model_clean.pkl")

# Sub-Scripts (Unified to use preprocess_lgbm.py)
CHATBOT_SCRIPT = os.path.join(SCRIPT_DIR, "run_chatbot.py")
PREPROCESS_SCRIPT = os.path.join(SCRIPT_DIR, "preprocess_lgbm.py")
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_lgbm.py")

sys.path.append(SCRIPT_DIR)

# Ensure Directories Exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =======================
# BACKEND TOOLS
# =======================
def verify_and_extract(disease_name, valid_symptoms):
    """Checks Wikipedia for disease existence and extracts symptoms."""
    print(f"   🔎 Verifying '{disease_name}' on Wikipedia...")
    try:
        url = f"https://en.wikipedia.org/wiki/{disease_name.replace(' ', '_')}"
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200: return None

        soup = BeautifulSoup(resp.text, 'html.parser')
        text = soup.get_text().lower()

        found = []
        for sym in valid_symptoms:
            if sym.replace("_", " ") in text:
                found.append(sym)

        if len(found) >= 1: return found
    except:
        pass
    return None

def fetch_and_save_advice(disease_name):
    if os.path.exists(ADVICE_DB_PATH):
        try:
            df = pd.read_csv(ADVICE_DB_PATH)
            if disease_name.lower() in df['Disease'].str.lower().values:
                print(f"   ⚠️ Advice for '{disease_name}' already exists.")
                return
        except:
            pass

    print(f"   🌍 Downloading precautions for '{disease_name}'...")
    tips = []
    source = "N/A"
    try:
        url = f"https://www.who.int/news-room/fact-sheets/detail/{disease_name.replace(' ', '-').lower()}"
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=3)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            for h in soup.find_all(['h2', 'h3']):
                if any(x in h.get_text() for x in ["Prevention", "Treatment"]):
                    for t in h.find_next_siblings(['p', 'ul'])[:3]:
                        clean = t.get_text().strip().replace("\n", " ")
                        if len(clean) > 20: tips.append(clean)
                    if tips: source = "WHO"; break
    except:
        pass

    if not tips:
        try:
            url = f"https://en.wikipedia.org/wiki/{disease_name.replace(' ', '_')}"
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for p in soup.find_all('p'):
                    if len(p.get_text()) > 50:
                        tips.append(f"Summary: {p.get_text().strip()[:200]}...");
                        source = "Wikipedia";
                        break
        except:
            pass

    if tips:
        while len(tips) < 3: tips.append("")
        with open(ADVICE_DB_PATH, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([disease_name.lower(), source, tips[0], tips[1], tips[2]])

def process_updates():
    print("\n[BACKEND] ⚙️  Checking for updates...")
    update_needed = False

    # 1. Process Requests
    if os.path.exists(REQUESTS_FILE) and os.path.exists(FEAT_PATH):
        df = pd.read_csv(REQUESTS_FILE)
        if 'status' not in df.columns: df['status'] = 'Pending'

        pending = df[df['status'] == 'Pending']

        if not pending.empty:
            print(f" [BACKEND] Processing {len(pending)} new requests...")

            unique_requests = pending['disease_name'].unique()
            existing_diseases = []
            if os.path.exists(LEARNED_DATA_FILE):
                try:
                    df_learned = pd.read_csv(LEARNED_DATA_FILE)
                    if 'prognosis' in df_learned.columns:
                        existing_diseases = df_learned['prognosis'].str.lower().unique().tolist()
                except:
                    pass

            known_symptoms = pd.read_csv(FEAT_PATH, nrows=0).columns.tolist()
            new_entries = []

            for d_name in unique_requests:
                clean_name = d_name.strip().lower()

                if clean_name in existing_diseases:
                    print(f"   ⚠️ SKIPPING '{d_name}': Already in database.")
                    df.loc[df['disease_name'] == d_name, 'status'] = 'Duplicate'
                    continue

                symptoms = verify_and_extract(d_name, known_symptoms)

                if symptoms:
                    print(f"   ✅ VERIFIED: '{d_name}'")
                    fetch_and_save_advice(d_name)
                    entry = {col: 0 for col in known_symptoms}
                    entry['prognosis'] = d_name.title()
                    for s in symptoms: entry[s] = 1
                    new_entries.append(entry)
                    df.loc[df['disease_name'] == d_name, 'status'] = 'Approved'
                    update_needed = True
                else:
                    print(f"   ❌ REJECTED: '{d_name}'")
                    df.loc[df['disease_name'] == d_name, 'status'] = 'Rejected'

            df.to_csv(REQUESTS_FILE, index=False)

            if new_entries:
                df_new = pd.DataFrame(new_entries)
                header = not os.path.exists(LEARNED_DATA_FILE)
                df_new.to_csv(LEARNED_DATA_FILE, mode='a', header=header, index=False)

    # 2. Check Timestamps
    if os.path.exists(LEARNED_DATA_FILE) and os.path.exists(MODEL_FILE):
        if os.path.getmtime(LEARNED_DATA_FILE) > os.path.getmtime(MODEL_FILE):
            update_needed = True

    # 3. Train
    if update_needed:
        print("\n [BACKEND] 🧠 Retraining Neural Model...")
        subprocess.run([sys.executable, PREPROCESS_SCRIPT], cwd=PROJECT_ROOT, check=True)
        subprocess.run([sys.executable, TRAIN_SCRIPT], cwd=PROJECT_ROOT, check=True)
        print(" [BACKEND] ✅ Update Complete.")
    else:
        print(" [BACKEND] System is up to date.")

# =======================
# MAIN EXECUTION
# =======================
def main():
    print("\n" + "#" * 60)
    print(" 🚀 AI PIPELINE: SYSTEM STARTUP")
    print("#" * 60)

    process_updates()

    print("\n" + "=" * 60)
    print(" 🏥 LAUNCHING CHATBOT INTERFACE...")
    print("=" * 60)

    if os.path.exists(CHATBOT_SCRIPT):
        subprocess.run([sys.executable, CHATBOT_SCRIPT], cwd=PROJECT_ROOT)
        print("\n[SYSTEM] Chatbot session ended.")
    else:
        print(f"❌ Error: {CHATBOT_SCRIPT} not found.")

if __name__ == "__main__":
    main()