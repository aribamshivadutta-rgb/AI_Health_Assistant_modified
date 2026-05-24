import pandas as pd
import json
import os

# Point to your Clean Data Folder
CLEAN_BASE = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System\data\clean\MedicalCRNN_clean"
TRAIN_EXCEL = os.path.join(CLEAN_BASE, 'Train_Label.csv') # Or .xlsx if you changed it back
TEST_EXCEL = os.path.join(CLEAN_BASE, 'Test_Label.csv')
SAVE_PATH = r"C:\Users\Bubu\AI-Healthcare-Diagnostic-System\models\medical_vocab.json"

unique_chars = set()
lexicon_set = set()

for file_path in [TRAIN_EXCEL, TEST_EXCEL]:
    if os.path.exists(file_path):
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        df['Text'].astype(str).apply(lambda x: unique_chars.update(list(x)))
        lexicon_set.update(df['Text'].astype(str).unique().tolist())

# Save to JSON
with open(SAVE_PATH, 'w', encoding='utf-8') as f:
    json.dump({
        "chars": sorted(list(unique_chars)),
        "lexicon": sorted(list(lexicon_set))
    }, f, ensure_ascii=False, indent=4)

print(f"✅ Created {SAVE_PATH} successfully!")