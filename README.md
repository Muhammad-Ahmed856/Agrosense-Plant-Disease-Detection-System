# 🌱 AgroSense v2.1 — AI Crop Intelligence Platform

An AI-powered plant disease detection web application built for farmers.
Combines a custom-trained CNN model and real-time weather advisory with user authentication.

**Built for:** Kirothon 2026 — AWS Community Pakistan  
**Institution:** Superior University, Lahore  
**Tech Stack:** Python, Flask, TensorFlow, SQLite, HTML/CSS/JS

---

## 📁 Project Structure

```
agrosense/
│
├── app.py                  ← Flask backend (all routes, auth, AI, weather)
├── train_model.py          ← CNN training script (run ONCE)
├── disease_info.py         ← Disease descriptions & remedies (38 classes)
├── requirements.txt        ← Python dependencies
├── agrosense.db            ← SQLite database (auto-created on first run)
├── README.md
│
├── model/
│   ├── plant_model.h5      ← Saved CNN model
│   └── class_names.json    ← 38 class labels
│
├── static/
│   └── uploads/            ← Uploaded leaf images
│
└── templates/
    ├── base.html           ← Shared sidebar layout
    ├── login.html          ← Login page (Farmer + Admin)
    ├── register.html       ← Farmer registration
    ├── index.html          ← Main dashboard (Diagnose, Weather, History)
    ├── admin.html          ← Admin panel
    ├── tips.html           ← Farming tips & guides
    ├── profile.html        ← User profile + password change
    └── about.html          ← About page + tech details
```

---

## ⚙️ Setup Instructions

### Step 1 — Install Dependencies

```bash
py -3.11 -m venv myenv
myenv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ Requires Python 3.9–3.11. TensorFlow does not yet support Python 3.12+.

### Step 2 — Setup Kaggle API (for dataset download)

1. Go to https://www.kaggle.com → Account → Create New API Token
2. Download `kaggle.json`
3. Place it at:
   - **Windows:** `C:\Users\YourName\.kaggle\kaggle.json`
   - **Linux/Mac:** `~/.kaggle/kaggle.json`

### Step 3 — Fix Windows Long Path Issue (Windows only)

The PlantVillage dataset has very long filenames. Run in PowerShell as Administrator:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Then restart your PC.

### Step 4 — Train the Model (Run ONCE — takes 1–3 hours)

```bash
python train_model.py
```

This will:
- Download the PlantVillage dataset (~2.7 GB via KaggleHub)
- Train for up to 15 epochs (EarlyStopping)
- Save `model/plant_model.h5` and `model/class_names.json`

> **Expected accuracy:** ~90–95% on validation set

### Step 5 — Run the Web App

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 👥 Default Accounts

| Role    | Username | Password  |
|---------|----------|-----------|
| Admin   | `admin`  | `admin123` |
| Farmer  | Register at `/register` | — |

> ⚠️ Change the admin password after first login via the Profile page.

---

## 🖥️ Pages & Features

### Farmer Pages

| Page | URL | Description |
|------|-----|-------------|
| Login | `/login` | Farmer or Admin login with role tabs |
| Register | `/register` | Create a new farmer account |
| Dashboard | `/dashboard` | Overview + quick actions |
| Disease Diagnosis | `/dashboard#diagnose` | Upload leaf → AI detection |
| Weather & Spray | `/dashboard#weather` | Live weather + spray advisory |
| Scan History | `/dashboard#history` | All previous scans with delete option |
| Farming Tips | `/tips` | Expert crop management guides |
| Profile | `/profile` | Edit name, phone, city + change password |
| About | `/about` | App info, model architecture, crop table |

### Admin Pages

| Page | URL | Description |
|------|-----|-------------|
| Admin Panel | `/admin` | Overview stats + user management + recent scans |
| Manage Tips | `/admin` → Tips tab | Add/delete farming tips |
| User Scans | Modal in admin | View all scans for any user |

---

## 🤖 AI Disease Detection Pipeline

The rejection system uses **3 stages** (OR logic — any stage passing = accepted):

```
Stage 0 — Color Heuristic (FAST)
  ✅ Checks for plant-like pixel colors:
     Green (healthy), Yellow (diseased), Brown (necrotic), Red (lesions)
  ✅ Threshold: 12% of pixels must be plant-colored
  ✅ Prevents false rejection of real leaves with unusual colors

Stage 1 — MobileNetV2 Semantic Check
  ✅ Pre-trained on ImageNet 1000 classes
  ✅ Sums probability of all plant-related classes
  ✅ Very lenient threshold (4%) — only rejects clearly non-plant images

→ Image accepted if Stage 0 OR Stage 1 passes

Stage 2 — CNN Disease Model Confidence
  ✅ Custom CNN trained on 87K PlantVillage images
  ✅ Minimum confidence: 30%
  ✅ Returns top-5 predictions with confidence scores
```

### Tuning Thresholds (in `app.py`)

```python
COLOR_PLANT_RATIO = 0.12   # Lower = more lenient color check
PLANT_SCORE_MIN   = 0.04   # Lower = more lenient semantic check
CONF_THRESHOLD    = 0.30   # Lower = accepts lower-confidence predictions
```

---

## 🌤️ Weather & Spray Advisory

Uses **Open-Meteo** (free, no API key needed). Evaluates:

| Factor | Ideal | Source |
|--------|-------|--------|
| 🌡️ Temperature | 15–30°C | IPM guidelines |
| 💧 Humidity | 40–90% | IPM guidelines |
| 🌬️ Wind Speed | < 15 km/h | Drift risk |
| 🌧️ Rain | 0 mm | Product washoff |

Verdict scale: **Excellent → Acceptable → Poor → Do NOT Spray**

---

## 🌾 Supported Crops (38 Disease Classes)

| Crop | Diseases |
|------|----------|
| 🍅 Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Target Spot, Yellow Leaf Curl, Mosaic Virus, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🌽 Corn | Gray Leaf Spot, Common Rust, Northern Blight, Healthy |
| 🍎 Apple | Scab, Black Rot, Cedar Rust, Healthy |
| 🍇 Grape | Black Rot, Esca, Leaf Blight, Healthy |
| 🍑 Peach | Bacterial Spot, Healthy |
| 🫑 Bell Pepper | Bacterial Spot, Healthy |
| 🍓 Strawberry | Leaf Scorch, Healthy |
| 🍊 Orange | Huanglongbing |
| 🍒 Cherry | Powdery Mildew, Healthy |
| Others | Blueberry, Raspberry, Soybean, Squash (Healthy / Powdery Mildew) |

---

## 📸 Tips for Best Scan Results

- Upload a **single leaf** filling most of the frame
- Use **natural daylight** — avoid flash or shadows
- Get **15–20 cm from the leaf** for best focus
- Keep the camera **steady** — avoid blur

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError` on Windows | Enable long paths (see Step 3) |
| Real leaves getting rejected | Lower `COLOR_PLANT_RATIO` to `0.08` or `CONF_THRESHOLD` to `0.20` |
| Model not loading | Run `python train_model.py` first |
| Kaggle download fails | Check `~/.kaggle/kaggle.json` is in the right place |
| `pip install` fails | Make sure Python 3.9–3.11 is used (not 3.12+) |

---

## 📦 Dependencies

```
flask>=2.3.0
tensorflow>=2.12.0
pillow>=9.0.0
numpy>=1.23.0
kagglehub>=0.2.0
requests>=2.28.0
opencv-python-headless>=4.7.0
```

Install all: `pip install -r requirements.txt`

---

*AgroSense — Protecting crops with the power of AI 🌱*
