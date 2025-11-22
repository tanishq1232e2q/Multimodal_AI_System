import os
os.environ["HF_HOME"] = "I:/" # Set Hugging Face cache directory
import zipfile
import gdown
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import tempfile
import pandas as pd
import numpy as np
import torch
import mne
import scipy.io
from torch import nn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import librosa
import torch.nn.functional as F

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def download_and_extract(drive_id: str, extract_to: str, expected_file: str = None):
    os.makedirs(extract_to, exist_ok=True)
    zip_path = os.path.join(extract_to, "dataset.zip")
    final_path = os.path.join(extract_to, expected_file) if expected_file else None

    # SKIP DOWNLOAD IF FINAL MODEL EXISTS
    if final_path and os.path.exists(final_path):
        print(f"Using cached: {final_path}")
        return

    # Otherwise download
    if not os.path.exists(zip_path):
        print(f"Downloading from Google Drive ID: {drive_id} ...")
        gdown.download(id=drive_id, output=zip_path, quiet=False)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)

    # DELETE ZIP
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"Cleaned up: {zip_path}")

    # Handle nested ZIPs
    for root, _, files in os.walk(extract_to):
        for f in files:
            if f.lower().endswith('.zip'):
                nested = os.path.join(root, f)
                print(f"Nested ZIP: {nested}")
                with zipfile.ZipFile(nested, 'r') as nz:
                    nz.extractall(root)
                os.remove(nested)

    # Final check
    if final_path and not os.path.exists(final_path):
        raise FileNotFoundError(f"Expected file not found: {final_path}")
    print(f"OK: {final_path or extract_to}")
# === DOWNLOAD ALL CHECKPOINTS ===
DATASET_URLS = {
    "spatial":        ("1W0vBdXZgVxyreh_CchyvEj6qTNivg_ds", "kaggle/working/spatial_model_checkpoint/model.pt"),
    "spatial-datas":  ("13g_HB5u_MuFZ7XMVaRwqRvuvsS9MZuju", None),
    "eeg":            ("1z1k1Om0jAPeaI1z9UvUR7xJamacjV690", "kaggle/working/eeg_model_checkpoint/model.pt"),
    # "eeg_stats":      ("1uZ29TcK0b8z2bmR0BMdcbW9iyrOfIn07", "eeg_preprocessing_stats.npz"),
    "textbert":       ("1RAqarb333OvrI9SV6VUcqOwkl7ytK970", "kaggle/working/textbert_model_checkpoint/model.pt"),
    "multilingual":   ("1e7NfhMLe9x2NS5SqHarbObikGC-YNv4E", "multilingual_my_model_checkpoint"),
    "audio":          ("1JlJ6Tkt-W89PjUkFSzxbxlk5xwy0YcPc", "kaggle/working/audio_model_checkpoint/model.pt"),
}

print("\n=== DOWNLOADING CHECKPOINTS ===")
for name, (drive_id, rel_path) in DATASET_URLS.items():
    sub_dir = os.path.join(CHECKPOINT_DIR, name)
    download_and_extract(drive_id, sub_dir, expected_file=rel_path)
print("=== ALL CHECKPOINTS READY ===\n")

# ---------- LOAD MODELS ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# AHRF Data
spatial_datas_dir = os.path.join(CHECKPOINT_DIR, "spatial-datas")
ahrf_paths = [
    os.path.join(spatial_datas_dir, "ahrf2023.csv"),
    os.path.join(spatial_datas_dir, "ahrf2024_Feb2025.csv")
]
years = [2023, 2024]
centroids_url = 'https://raw.githubusercontent.com/davidingold/county_centroids/master/county_centroids.csv'
centroids_df = pd.read_csv(centroids_url)

geo_cols = ['fips_st_cnty', 'st_name', 'cnty_name']
mh_keywords = ['povty', 'emplymt', 'eductn', 'psych', 'phys_nf_prim_care']
target_col_base = 'hi_povty_typolgy_14'

merged_df = []
for i, path in enumerate(ahrf_paths):
    print(f"Loading {years[i]} from {path}")
    df_year = pd.read_csv(path, encoding='latin1', low_memory=False)
    mh_related_cols_year = [col for col in df_year.columns if any(kw in col.lower() for kw in mh_keywords)]
    df_selected = df_year[geo_cols + mh_related_cols_year].copy()
    df_selected['fips_st_cnty'] = df_selected['fips_st_cnty'].astype(str).str.zfill(5)
    target_col_year = target_col_base
    if target_col_year not in df_selected.columns:
        pov_cols = [col for col in df_selected.columns if 'povty' in col.lower()]
        if pov_cols:
            target_col_year = pov_cols[0]
            print(f"Using fallback poverty col '{target_col_year}' for {years[i]}")
    df_selected['high_mh_disorder'] = (pd.to_numeric(df_selected[target_col_year], errors='coerce') > 0).astype(int)
    df_selected = df_selected.dropna(subset=['high_mh_disorder'])
    df_selected['year'] = years[i]
    df_selected['fips_st_cnty'] = df_selected['fips_st_cnty'].astype(str).str.zfill(5)
    centroids_df['GEOID'] = centroids_df['GEOID'].astype(str).str.zfill(5)
    df_merged_year = pd.merge(df_selected, centroids_df[['GEOID', 'x', 'y']], left_on='fips_st_cnty', right_on='GEOID', how='left')
    merged_df.append(df_merged_year)
    print(f"{years[i]} shape after merge: {df_merged_year.shape}")

if not merged_df:
    raise ValueError("No AHRF data loaded")

merged_all_years = pd.concat(merged_df, ignore_index=True)
feature_cols = [col for col in merged_all_years.columns if col not in ['fips_st_cnty', 'st_name', 'cnty_name', 'GEOID', 'x', 'y', 'high_mh_disorder', 'year']]
scaler = StandardScaler()
scaler.fit(merged_all_years[feature_cols].values)

# ---------- MODELS ----------
unified_classes = ['normal', 'adhd', 'anxiety', 'bipolar', 'depression']
text_classes = ['adhd', 'anxiety', 'autism', 'bipolar', 'bpd', 'depression', 'ptsd', 'schizophrenia']

# TextBERT
tokenizer_text = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model_text = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=8)
textbert_path = os.path.join(CHECKPOINT_DIR, "textbert", "kaggle/working/textbert_model_checkpoint/model.pt")
model_text.load_state_dict(torch.load(textbert_path, map_location=device))
model_text.to(device)
model_text.eval()

# Multilingual
multilingual_dir = os.path.join(CHECKPOINT_DIR, "multilingual", "multilingual_my_model_checkpoint")
tokenizer_multilingual = AutoTokenizer.from_pretrained(multilingual_dir)
model_multilingual = AutoModelForSequenceClassification.from_pretrained(multilingual_dir)
model_multilingual.to(device)
model_multilingual.eval()

# EEG
num_channels = 129 * 5
class CNNGRU(nn.Module):
    def __init__(self, num_channels=num_channels, num_classes=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(num_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
        )
        self.gru = nn.GRU(512, 256, num_layers=3, batch_first=True, dropout=0.4)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = self.cnn(x)
        x = x.transpose(1, 2)
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])

model_eeg = CNNGRU().to(device)
eeg_path = os.path.join(CHECKPOINT_DIR, "eeg", "kaggle/working/eeg_model_checkpoint/model.pt")
model_eeg.load_state_dict(torch.load(eeg_path, map_location=device))
model_eeg.eval()

# EEG stats
preproc_npz = "./checkpoints/eeg_preprocessing_stats.npz"  # MUST UPLOAD THIS!
stats = np.load(preproc_npz)
EEG_MEAN, EEG_STD = stats['mean'], stats['std']
MEAN_STD = (EEG_MEAN, EEG_STD)

# Audio
class GRUClassifier(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=3, num_classes=3, dropout_rate=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.bn(out[:, -1, :])
        out = self.fc(out)
        return out

model_audio = GRUClassifier().to(device)
audio_path = os.path.join(CHECKPOINT_DIR, "audio", "kaggle/working/audio_model_checkpoint/model.pt")
model_audio.load_state_dict(torch.load(audio_path, map_location=device))
model_audio.eval()

# Spatial
class Spatial1DCNN(nn.Module):
    def __init__(self, input_length=272):
        super().__init__()
        self.input_length = input_length
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        final_length = ((input_length + 1) // 2 + 1) // 2
        self.fc1 = nn.Linear(64 * final_length, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

model_spatial = Spatial1DCNN().to(device)
spatial_path = os.path.join(CHECKPOINT_DIR, "spatial", "kaggle/working/spatial_model_checkpoint/model.pt")
model_spatial.load_state_dict(torch.load(spatial_path, map_location=device))
model_spatial.eval()


class FusionMLP(nn.Module):
    def __init__(self, input_size=40, num_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, features):
        x = self.fc1(features)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

fusion_model = FusionMLP().to(device)
fusion_model.eval()

# ---------- FUNCTIONS ----------
def extract_psd_features_from_mat(mat_path, num_channels=129, sfreq=250, bands=None, mean_std=None, device='cpu'):
    if bands is None:
        bands = [(0.5,4), (4,8), (8,13), (13,30), (30,50)]
    mat = scipy.io.loadmat(mat_path)
    sys_keys = {'__header__', '__version__', '__globals__', 'samplingRate', 'Impedances_0', 'DIN_1'}
    data_key = next((k for k in mat.keys() if k not in sys_keys), None)
    if data_key is None:
        raise ValueError(f"No EEG data in {mat_path}")
    eeg = mat[data_key]
    if eeg.ndim > 2: eeg = np.squeeze(eeg)
    if eeg.ndim != 2: raise ValueError(f"EEG must be 2D")
    if eeg.shape[1] == num_channels: eeg = eeg.T
    ch_names = [f"ch{i+1}" for i in range(num_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(eeg, info, verbose=False)
    band_psds = []
    for fmin, fmax in bands:
        psd, _ = mne.time_frequency.psd_array_welch(raw.get_data(), sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=256, verbose=False)
        band_psds.append(psd.mean(axis=1))
    features = np.concatenate(band_psds)
    if mean_std is not None:
        mean, std = mean_std
        features = (features - mean) / (std + 1e-8)
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    return tensor

def get_spatial_input_from_fips(fips_code, merged_df, scaler, feature_cols):
    
    
    fips_str = str(fips_code).zfill(5)

    # Select matching row
    row = merged_df[merged_df['fips_st_cnty'] == fips_str]
    if row.empty:
        raise ValueError(f"FIPS code {fips_str} not found in AHRF dataset!")

    # Extract and scale features
    spatial_vector = row.iloc[0][feature_cols].values.astype(np.float32)
    scaled = scaler.transform(spatial_vector.reshape(1, -1))[0]

    # Replace NaN, inf, or -inf with 0.0 for JSON safety
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    return scaled.tolist()




def predict_all_modalities(text=None, multilingual_text=None, audio_input=None, eeg_input=None, spatial_input=None, sr=16000):
    results = {'individual_predictions': {}}
    all_logits = []

    # Text
    if text:
        # inputs_text = tokenizer_text(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        # with torch.no_grad():
        #     logits_text = model_text(**inputs_text).logits
        # all_logits.append(logits_text)
        # pred_text = text_classes[torch.argmax(logits_text, dim=1).item()]
        # results['text'] = pred_text
        # === TEXT ANALYSIS (Adaptive Normal Detection) ===
        inputs_text = tokenizer_text(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(device)

        with torch.no_grad():
            logits_text = model_text(**inputs_text).logits

        # Convert logits → probabilities for each class
        probs_text = torch.softmax(logits_text, dim=1).cpu().numpy()[0]

        # Create dictionary of probabilities for all classes
        text_probs_dict = {
            text_classes[i]: round(float(probs_text[i]), 2)
            for i in range(len(text_classes))
        }

        # Get top class and score
        pred_idx = int(np.argmax(probs_text))
        pred_label = text_classes[pred_idx]
        pred_score = float(probs_text[pred_idx])

        # Adaptive "Normal" detection (optional)
        # confidence_threshold = 0.55
        # if pred_score < confidence_threshold:
        #     pred_label = "Normal"
        if pred_label=="depression" and pred_score < 0.65:
            pred_label = "Normal"


        # Save results
        results['text'] = pred_label
        results['individual_predictions']['text_score'] = round(pred_score, 2)
        results['text_probabilities'] = text_probs_dict

    # Multilingual
    if text or multilingual_text:
        input_text = multilingual_text if multilingual_text else text
        inputs = tokenizer_multilingual(
            input_text,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            logits = model_multilingual(**inputs).logits

        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_label = ['Normal', 'Depression'][pred_idx]
        results['multilingual'] = pred_label
    else:
        results['multilingual'] = 'N/A'

    # Audio
    if audio_input:
        y, sr = librosa.load(audio_input, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        max_pad_len = 100
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        mfcc = mfcc.T
        audio_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_audio = model_audio(audio_tensor)
        all_logits.append(logits_audio)
        pred_audio = ['normal', 'mild_depression', 'severe_depression'][torch.argmax(logits_audio, dim=1).item()]
        results['audio'] = pred_audio
        # results['individual_predictions']['audio_score'] = float(pred_audio)
    else:
        all_logits.append(torch.zeros(1, 3).to(device))
        results['audio'] = 'N/A'

    # EEG
    if eeg_input:
        try:
            eeg_tensor = extract_psd_features_from_mat(eeg_input, mean_std=MEAN_STD, device=device)
            with torch.no_grad():
                logits_eeg = model_eeg(eeg_tensor)
            all_logits.append(logits_eeg)
            probs = torch.softmax(logits_eeg, dim=1).cpu().numpy()[0]
            pred_eeg = 'MDD(Major Depressive Disorder)' if probs[1] > 0.5 else 'HC(Healthy Control)'
            results['eeg'] = pred_eeg
            results['individual_predictions']['eeg_score'] = float(probs[1])
        except Exception as e:
            print(f"EEG error: {e}")
            all_logits.append(torch.zeros(1, 2).to(device))
            results['eeg'] = 'N/A'
            results['individual_predictions']['eeg_score'] = 0.0
    else:
        all_logits.append(torch.zeros(1, 2).to(device))
        results['eeg'] = 'N/A'
        results['individual_predictions']['eeg_score'] = 0.0

    # Spatial
    if spatial_input is not None:
        spatial_array = np.asarray(spatial_input, dtype=np.float32)
        expected_length = 272
        if spatial_array.shape[0] != expected_length:
            if spatial_array.shape[0] < expected_length:
                spatial_array = np.pad(spatial_array, (0, expected_length - spatial_array.shape[0]), mode='constant')
            else:
                spatial_array = spatial_array[:expected_length]
        spatial_tensor = torch.tensor(spatial_array, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
        with torch.no_grad():
            logits_spatial = model_spatial(spatial_tensor)
        all_logits.append(logits_spatial)
        spatial_prob = torch.sigmoid(logits_spatial).item()
        results['individual_predictions']['spatial_score'] = spatial_prob
        results['individual_predictions']['spatial_risk'] = 'HighRisk' if spatial_prob > 0.51 else 'LowRisk'
    else:
        all_logits.append(torch.zeros(1, 1).to(device))
        results['individual_predictions']['spatial_risk'] = 'N/A'

    return results

print("All models loaded and ready.")



@app.post("/predict")
async def predict(
    text: str = Form(None),
    multilingual_text: str = Form(None),
    audio: UploadFile = File(None),
    eeg: UploadFile = File(None),
    fips_code: str = Form(None)
):
    try:
        audio_path = None
        eeg_path = None
        spatial_input = None

        if audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                shutil.copyfileobj(audio.file, tmp)
                audio_path = tmp.name

        if eeg:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
                shutil.copyfileobj(eeg.file, tmp)
                eeg_path = tmp.name

        if fips_code:
            spatial_input = get_spatial_input_from_fips(fips_code, merged_all_years, scaler, feature_cols)

        result = predict_all_modalities(text, multilingual_text, audio_path, eeg_path, spatial_input)

        # CLEAN UP
        if audio_path: os.unlink(audio_path)
        if eeg_path: os.unlink(eeg_path)

        # BUILD RESPONSE
        individual = {
            'text': result.get('text', 'N/A'),
            "text_score": result.get("individual_predictions", {}).get("text_score", 0),
            'multilingual': result.get('multilingual', 'N/A'),
            'audio': result.get('audio', 'N/A'),
            # 'audio_score':result['individual_predictions'].get('audio_score', 'N/A'),
            'eeg': result.get('eeg', 'N/A'),
            'eeg_score': round(float(result['individual_predictions'].get('eeg_score', 0.0)),2),
            'spatial_risk': result['individual_predictions'].get('spatial_risk', 'N/A'),
            'spatial_score': round(float(result['individual_predictions'].get('spatial_score', 0.0)),2),
        }
        print(individual)

        return {"predictions": individual}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000,reload=False)