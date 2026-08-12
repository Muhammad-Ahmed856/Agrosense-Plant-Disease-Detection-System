"""
AgroSense v2.1 - Flask Web Application
=======================================
Features:
  - Farmer login / register / profile
  - Admin panel (manage users, view all scans)
  - Scan history per user (SQLite)
  - 2-stage non-plant rejection with fallback color heuristic
  - Weather & Pesticide Advisory (Open-Meteo, no API key)
  - Tips, About, Contact pages
"""

import os, json, uuid, requests, sqlite3, hashlib, secrets
import numpy as np
from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for, g)
from PIL import Image
import io
from functools import wraps

# ── Config ─────────────────────────────────────────────────────
_MODEL_H5     = os.path.join("model", "plant_model.h5")
_MODEL_KERAS  = os.path.join("model", "plant_model.keras")
# Prefer .keras (modern format, no compatibility issues), fall back to .h5
MODEL_PATH    = _MODEL_KERAS if os.path.exists(_MODEL_KERAS) else _MODEL_H5
NAMES_PATH    = os.path.join("model", "class_names.json")
UPLOAD_FOLDER = os.path.join("static", "uploads")
DB_PATH       = "agrosense.db"
ALLOWED_EXT   = {"png", "jpg", "jpeg", "webp", "bmp"}
IMG_SIZE      = (224, 224)   # FIXED: must match train_model.py — MobileNetV2 native size
MAX_FILE_SIZE = 10 * 1024 * 1024

# ── Rejection thresholds (tuned to avoid false rejections) ──────
# Stage 0: HSV color heuristic — passes if enough green/yellow/brown pixels
COLOR_PLANT_RATIO = 0.25   # 25% of pixels must look like plant material (stricter)

# Stage 1: MobileNetV2 — sum of plant-related ImageNet class probabilities
PLANT_SCORE_MIN   = 0.15   # stricter — reduce false positives on green non-plant images

# Stage 2: Disease CNN confidence
CONF_THRESHOLD    = 0.40   # increased from 0.30 — minimum confidence to accept prediction

# ── Expanded plant-related ImageNet class indices ───────────────
# Covers: flowers, fruits, vegetables, trees, fungi, leaves, grass, crops
# Sources: ImageNet class list + PlantVillage-adjacent categories
PLANT_IMAGENET_INDICES = set([
    # Flowers (985–999 = daisy, dandelion, roses, sunflowers, tulips etc.)
    985,986,987,988,989,990,991,992,993,994,995,996,997,998,999,
    # Fruits & berries
    948,949,950,951,952,953,954,955,956,957,958,959,960,961,962,963,964,
    # Vegetables / food plants
    936,937,938,939,940,941,942,943,944,945,946,947,
    # Trees (oak, palm, pine, etc.)
    340,341,342,343,344,345,346,347,348,349,350,351,352,
    # Fungi / mushrooms (appear on diseased plants)
    281,282,283,
    # Corn / maize
    987,
    # Cabbage / broccoli family
    936,
    # Fern, lichen, moss-like
    992,993,
    # Pot plant / house plant
    727,
    # Leaf-adjacent: green background, nature scenes
    # (not strict plants but high co-occurrence with leaf photos)
    334,335,336,337,338,339,
    # Strawberry
    949,
    # Grapes
    951,
    # Apple / pear
    948,950,
    # Pepper / chili
    942,
    # Banana leaf (broad leaf crops)
    954,
    # Artichoke
    947,
])

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ══════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db: db.close()

def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT UNIQUE NOT NULL,
            email     TEXT UNIQUE NOT NULL,
            password  TEXT NOT NULL,
            role      TEXT NOT NULL DEFAULT 'farmer',
            full_name TEXT DEFAULT '',
            phone     TEXT DEFAULT '',
            city      TEXT DEFAULT '',
            active    INTEGER DEFAULT 1,
            created   TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS scans (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            image_url   TEXT,
            plant       TEXT,
            disease     TEXT,
            severity    TEXT,
            confidence  REAL,
            remedy      TEXT,
            is_plant    INTEGER DEFAULT 1,
            scanned_at  TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS tips (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            title      TEXT NOT NULL,
            content    TEXT NOT NULL,
            category   TEXT DEFAULT 'general',
            created_by INTEGER,
            created    TEXT DEFAULT (datetime('now'))
        );
    """)
    cur = db.execute("SELECT id FROM users WHERE username='admin'")
    if not cur.fetchone():
        db.execute("""INSERT INTO users (username,email,password,role,full_name)
                      VALUES ('admin','admin@agrosense.pk',?,'admin','Administrator')""",
                   (hash_password("admin123"),))
    # Seed some tips
    cur2 = db.execute("SELECT COUNT(*) FROM tips")
    if cur2.fetchone()[0] == 0:
        tips = [
            ("Best Time to Spray Pesticide","Always spray pesticides in early morning (6–9 AM) or late evening. Avoid spraying during hot midday — it causes evaporation and leaf burn.","spray"),
            ("How to Identify Early Blight","Look for dark, concentric ring spots (like a target board) on lower, older leaves first. Remove affected leaves and apply mancozeb or chlorothalonil.","disease"),
            ("Preventing Late Blight","Late Blight spreads rapidly in wet weather. Avoid overhead irrigation. Use certified disease-free seeds and apply metalaxyl preventively.","disease"),
            ("Water Management Tips","Water at the base of plants, not on leaves. Morning watering is best — leaves dry before evening, reducing fungal risk.","general"),
            ("Soil Health for Better Crops","Test your soil every season. Balanced NPK fertilization prevents nutrient deficiencies that make plants more susceptible to disease.","general"),
            ("Crop Rotation Benefits","Rotate crops every season to break pest and disease cycles. Never plant tomatoes in the same spot two years in a row.","general"),
        ]
        for t in tips:
            db.execute("INSERT INTO tips (title,content,category) VALUES (?,?,?)", t)
    db.commit()
    db.close()

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

init_db()

# ── Auth helpers ───────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def dec(*a, **kw):
        if "user_id" not in session: return redirect(url_for("login_page"))
        return f(*a, **kw)
    return dec

def admin_required(f):
    @wraps(f)
    def dec(*a, **kw):
        if "user_id" not in session or session.get("role") != "admin":
            return redirect(url_for("login_page"))
        return f(*a, **kw)
    return dec

# ══════════════════════════════════════════════════════════════
#  AI MODELS
# ══════════════════════════════════════════════════════════════
disease_model = None
gatekeeper    = None
class_names   = []

def load_models():
    global disease_model, gatekeeper, class_names
    import tensorflow as tf
    if not os.path.exists(MODEL_PATH):
        print("WARNING: Disease model not found at", MODEL_PATH)
        print("         → Run: python train_model.py")
        return False
    if not os.path.exists(NAMES_PATH):
        print("WARNING: class_names.json not found at", NAMES_PATH)
        print("         → Run: python train_model.py")
        return False
    
    print(f"Loading disease model from: {MODEL_PATH}")

    # ── Pre-flight: detect old/incompatible model format ──────────────────
    try:
        import h5py, json as _json
        with h5py.File(MODEL_PATH, "r") as _f:
            _kv  = _f.attrs.get("keras_version", b"unknown")
            _cfg = _json.loads(_f.attrs.get("model_config", "{}"))
            _layers = _cfg.get("config", {}).get("layers", [])
            # Check ALL layer names for mobilenet (not just first layer)
            _all_names = " ".join(
                l.get("config", {}).get("name", "") for l in _layers
            ).lower()
        if "mobilenet" not in _all_names:
            print(f"❌ INCOMPATIBLE MODEL DETECTED")
            print(f"   Saved with Keras {_kv} — no MobileNetV2 backbone found.")
            print(f"   Your plant_model.h5 is an OLD custom CNN — it must be replaced.")
            print(f"   ╔══════════════════════════════════════════════════════╗")
            print(f"   ║  DELETE model/plant_model.h5                        ║")
            print(f"   ║  Then run:  python train_model.py                   ║")
            print(f"   ║  Or train on Kaggle (recommended, ~15 min on GPU)   ║")
            print(f"   ╚══════════════════════════════════════════════════════╝")
            return False
        print(f"✅ Pre-flight check passed — MobileNetV2 backbone confirmed")
    except Exception:
        pass  # h5py not available — proceed and let keras report the error
    # ──────────────────────────────────────────────────────────────────────

    try:
        disease_model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects=None
        )
        with open(NAMES_PATH) as f:
            class_names = json.load(f)
        print(f"✅ Disease model loaded — {len(class_names)} classes")
        print(f"   Model type: {type(disease_model).__name__}")
        print(f"   Model input shape: {disease_model.input_shape}")
    except Exception as e:
        print(f"❌ ERROR loading model: {type(e).__name__}: {e}")
        print(f"   → Fix: Delete model/plant_model.h5 and run: python train_model.py")
        return False
    
    try:
        print("Loading MobileNetV2 gatekeeper...")
        gatekeeper = tf.keras.applications.MobileNetV2(
            weights="imagenet", include_top=True, input_shape=(224,224,3))
        print("✅ MobileNetV2 gatekeeper loaded")
    except Exception as e:
        print(f"⚠️  Gatekeeper unavailable: {e}")
        gatekeeper = None
    
    return True

model_ready = load_models()

# ── Stage 0: HSV Color Heuristic + Texture Detection ──────────────────
def color_plant_check(pil_img):
    """
    Improved leaf detection: checks for plant colors AND leaf-like texture/edges.
    Prevents false positives from solid green images.
    
    Returns (passed, ratio, reason)
    """
    import cv2
    
    rgb = np.array(pil_img.convert("RGB").resize((128,128)), dtype=np.float32)
    R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    M = np.maximum.reduce([R,G,B])
    m = np.minimum.reduce([R,G,B])

    # Green pixels: G dominates significantly
    green  = (G > R + 15) & (G > B + 15) & (G > 40)
    # Yellow: R and G both high, B low
    yellow = (R > 120) & (G > 100) & (B < 100) & (np.abs(R.astype(int)-G.astype(int)) < 60)
    # Brown/tan: moderate R, lower G, low B
    brown  = (R > 80) & (R > G + 5) & (G > 40) & (B < 120) & (M - m < 120)
    # Red (disease spots): R much higher than G and B
    red    = (R > 100) & (R > G + 30) & (R > B + 30)

    plant_mask = green | yellow | brown | red
    color_ratio = float(plant_mask.sum()) / float(plant_mask.size)
    
    # If color ratio fails, reject immediately
    if color_ratio < COLOR_PLANT_RATIO:
        return False, color_ratio, f"Insufficient plant-like colors ({color_ratio*100:.1f}%)"
    
    # Additional texture check: look for edges/veins typical of leaves
    # Convert to grayscale and compute edge density
    gray = np.array(pil_img.convert("L").resize((128,128)), dtype=np.uint8)
    
    # Compute Laplacian (edge detection)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    edge_energy = np.mean(np.abs(laplacian))
    
    # Solid color images have very low edge energy
    # Real leaves have moderate-to-high edge energy due to veins and texture
    if edge_energy < 2.0:  # too smooth, probably solid green color
        return False, color_ratio, f"Image too smooth ({edge_energy:.2f}) — no leaf texture detected"
    
    # All checks passed
    return True, color_ratio, "OK"

# ── Stage 1: MobileNetV2 Gatekeeper ───────────────────────────
def gate_check(pil_img):
    if gatekeeper is None: return True, 1.0, "gatekeeper_unavailable"
    import tensorflow as tf
    img  = pil_img.convert("RGB").resize((224,224))
    arr  = tf.keras.applications.mobilenet_v2.preprocess_input(
               np.array(img, dtype=np.float32))
    preds = gatekeeper.predict(np.expand_dims(arr,0), verbose=0)[0]
    plant_score = float(sum(preds[i] for i in PLANT_IMAGENET_INDICES if i < len(preds)))
    decoded   = tf.keras.applications.mobilenet_v2.decode_predictions(preds[np.newaxis], top=1)
    top_label = decoded[0][0][1].replace("_"," ") if decoded else "unknown"
    return plant_score >= PLANT_SCORE_MIN, plant_score, top_label

# ── Stage 2: Disease CNN ───────────────────────────────────────
def disease_check(pil_img):
    """
    Run the disease classification model.

    IMPORTANT: The model was trained with MobileNetV2 using
    mobilenet_v2.preprocess_input(), which scales pixels to [-1, 1].
    We must use the SAME preprocessing here — using /255.0 instead
    would produce completely wrong predictions, especially on
    real-world (Google) images.
    """
    import tensorflow as tf
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    # MobileNetV2 preprocessing: scales [0,255] → [-1, 1]
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, 0)
    preds = disease_model.predict(arr, verbose=0)[0]
    idx   = preds.argsort()[::-1]
    top5  = [(class_names[i], float(preds[i])) for i in idx[:5]]
    return class_names[idx[0]], float(preds[idx[0]]), top5, float(preds[idx[0]]) >= CONF_THRESHOLD

def predict_image(pil_img):
    """
    3-stage pipeline:
      Stage 0 — Color heuristic + texture detection (fast, catches real leaves)
      Stage 1 — MobileNetV2 semantic check (only runs if Stage 0 fails)
      Stage 2 — Disease model confidence check
    
    KEY: Stage 0 OR Stage 1 must pass (they are OR'd, not AND'd).
    This means a real leaf will pass even if MobileNetV2 misclassifies it,
    as long as it has plant-like colors AND leaf texture.
    """
    # Always run disease check (we need top5 regardless)
    best_name, best_conf, top5, s2_pass = disease_check(pil_img)

    # Stage 0: color heuristic + texture detection
    color_pass, color_ratio, color_reason = color_plant_check(pil_img)

    # Stage 1: semantic gatekeeper
    s1_pass, plant_score, top_label = gate_check(pil_img)

    # Accept if EITHER color OR semantic check passes
    visual_ok = color_pass or s1_pass

    if not visual_ok:
        return {
            "is_plant": False, "stage": 0,
            "class_name": best_name, "confidence": best_conf,
            "top5": top5, "plant_score": plant_score, "top_label": top_label,
            "color_ratio": color_ratio,
            "rejection_reason": (
                f'This image does not appear to contain a plant leaf. '
                f'Content detector: "{top_label}" (plant score {plant_score*100:.1f}%), '
                f'Plant colors: {color_ratio*100:.1f}% ({color_reason}). '
                f'Please upload a clear, close-up photo of a crop leaf.'
            )
        }

    if not s2_pass:
        return {
            "is_plant": False, "stage": 2,
            "class_name": best_name, "confidence": best_conf,
            "top5": top5, "plant_score": plant_score, "top_label": top_label,
            "color_ratio": color_ratio,
            "rejection_reason": (
                f"A plant was detected but the disease model is uncertain "
                f"({best_conf*100:.1f}% confidence — minimum {CONF_THRESHOLD*100:.0f}% required). "
                f"Try a clearer, closer, better-lit photo of the leaf."
            )
        }

    return {
        "is_plant": True, "stage": None,
        "class_name": best_name, "confidence": best_conf,
        "top5": top5, "plant_score": plant_score, "top_label": top_label,
        "color_ratio": color_ratio, "rejection_reason": ""
    }

# ══════════════════════════════════════════════════════════════
#  WEATHER
# ══════════════════════════════════════════════════════════════
def get_pesticide_advisory(temp, humidity, wind_kmh, rain_mm, weather_code):
    good, bad = [], []
    score = 100
    if   temp < 5:   bad.append(f"Too cold ({temp}°C) — pesticide won't activate"); score-=40
    elif temp < 15:  bad.append(f"Low temp ({temp}°C) — reduced absorption, apply cautiously"); score-=15
    elif temp <= 30: good.append(f"Ideal temperature ({temp}°C) — excellent absorption")
    elif temp > 35:  bad.append(f"Very hot ({temp}°C) — leaf burn risk, rapid evaporation"); score-=30
    else:            bad.append(f"Warm ({temp}°C) — apply early morning to minimize evaporation"); score-=10
    if   humidity < 30:  bad.append(f"Very dry ({humidity}%) — spray evaporates before absorbing"); score-=25
    elif humidity < 40:  bad.append(f"Low humidity ({humidity}%) — slightly reduced effectiveness"); score-=10
    elif humidity <= 90: good.append(f"Good humidity ({humidity}%) — spray will adhere well to leaves")
    else:                bad.append(f"Very humid ({humidity}%) — spray may wash off, disease pressure high"); score-=10
    if   wind_kmh > 25: bad.append(f"Strong wind ({wind_kmh:.0f} km/h) — dangerous drift, do NOT spray"); score-=50
    elif wind_kmh > 15: bad.append(f"Moderate wind ({wind_kmh:.0f} km/h) — spray drift risk, use caution"); score-=20
    else:               good.append(f"Calm wind ({wind_kmh:.0f} km/h) — no drift risk, ideal for spraying")
    if   rain_mm > 5: bad.append(f"Rain {rain_mm:.1f}mm — will wash pesticide off leaves, do NOT spray"); score-=60
    elif rain_mm > 1: bad.append(f"Light rain {rain_mm:.1f}mm — may reduce effectiveness"); score-=20
    else:             good.append("No rain — pesticide will remain on leaf surface")
    if weather_code in range(51,68) or weather_code in range(80,83):
        bad.append("Active rain/drizzle — delay spraying until dry"); score-=40
    elif weather_code in range(95,100):
        bad.append("Thunderstorm — do not spray, dangerous conditions"); score-=100
    score = max(0, min(100, score))
    if   score >= 75: v,ve,ae="excellent","✅ Excellent Time to Spray","All conditions are ideal. Spray now for maximum effectiveness. Early morning (6–9 AM) is best."
    elif score >= 50: v,ve,ae="good","🟡 Acceptable — Apply with Care","Conditions are acceptable. Some factors need attention — see breakdown below."
    elif score >= 25: v,ve,ae="poor","🔴 Poor Conditions — Wait if Possible","Multiple unfavorable conditions. Spraying now risks wasting pesticide or harming crops."
    else:             v,ve,ae="danger","🚫 Do NOT Spray Today","Conditions are dangerous. Risk of plant damage, spray drift, or complete product washoff."
    return {"score":score,"verdict":v,"verdict_en":ve,"advice_en":ae,"good":good,"bad":bad}

def wmo_desc(code):
    m={0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",
       45:"Foggy",48:"Icy fog",51:"Light drizzle",53:"Moderate drizzle",
       61:"Slight rain",63:"Moderate rain",65:"Heavy rain",
       71:"Light snow",73:"Moderate snow",75:"Heavy snow",
       80:"Slight showers",81:"Moderate showers",82:"Heavy showers",
       95:"Thunderstorm",96:"Thunderstorm + hail"}
    return m.get(code, f"Code {code}")

def allowed_file(f):
    return "." in f and f.rsplit(".",1)[1].lower() in ALLOWED_EXT

# ══════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ══════════════════════════════════════════════════════════════
@app.route("/")
def root():
    return redirect(url_for("dashboard") if "user_id" in session else url_for("login_page"))

@app.route("/login")
def login_page():
    if "user_id" in session: return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route("/api/login", methods=["POST"])
def api_login():
    d = request.get_json()
    u, p = d.get("username","").strip(), d.get("password","")
    if not u or not p: return jsonify({"error":"Username and password required"}), 400
    db  = get_db()
    row = db.execute("SELECT * FROM users WHERE username=? AND active=1",(u,)).fetchone()
    if not row or row["password"] != hash_password(p):
        return jsonify({"error":"Invalid username or password"}), 401
    session.update({"user_id":row["id"],"username":row["username"],
                    "role":row["role"],"full_name":row["full_name"] or row["username"]})
    return jsonify({"success":True,"role":row["role"]})

@app.route("/api/register", methods=["POST"])
def api_register():
    d = request.get_json()
    un,em,pw = d.get("username","").strip(),d.get("email","").strip(),d.get("password","")
    fn,ph,ci = d.get("full_name","").strip(),d.get("phone","").strip(),d.get("city","").strip()
    if not un or not em or not pw: return jsonify({"error":"Username, email and password required"}), 400
    if len(pw) < 6: return jsonify({"error":"Password must be at least 6 characters"}), 400
    try:
        db = get_db()
        db.execute("INSERT INTO users (username,email,password,full_name,phone,city) VALUES (?,?,?,?,?,?)",
                   (un,em,hash_password(pw),fn,ph,ci))
        db.commit()
        return jsonify({"success":True})
    except sqlite3.IntegrityError:
        return jsonify({"error":"Username or email already exists"}), 409

@app.route("/logout")
def logout():
    session.clear(); return redirect(url_for("login_page"))

# ══════════════════════════════════════════════════════════════
#  MAIN APP ROUTES
# ══════════════════════════════════════════════════════════════
@app.route("/dashboard")
@login_required
def dashboard():
    db = get_db()
    scan_count = db.execute("SELECT COUNT(*) FROM scans WHERE user_id=?",(session["user_id"],)).fetchone()[0]
    return render_template("index.html", model_ready=model_ready,
        username=session["username"], full_name=session["full_name"],
        role=session["role"], scan_count=scan_count)

@app.route("/tips")
@login_required
def tips_page():
    db   = get_db()
    tips = db.execute("SELECT * FROM tips ORDER BY created DESC").fetchall()
    return render_template("tips.html", tips=[dict(t) for t in tips],
        model_ready=model_ready,
        username=session["username"], full_name=session["full_name"], role=session["role"])

@app.route("/profile")
@login_required
def profile_page():
    db  = get_db()
    row = db.execute("SELECT * FROM users WHERE id=?",(session["user_id"],)).fetchone()
    sc  = db.execute("SELECT COUNT(*) FROM scans WHERE user_id=? AND is_plant=1",(session["user_id"],)).fetchone()[0]
    rc  = db.execute("SELECT COUNT(*) FROM scans WHERE user_id=?",(session["user_id"],)).fetchone()[0]
    return render_template("profile.html", user=dict(row), scan_count=sc, total_count=rc,
        model_ready=model_ready,
        username=session["username"], full_name=session["full_name"], role=session["role"])

@app.route("/api/profile", methods=["POST"])
@login_required
def api_update_profile():
    d  = request.get_json()
    fn = d.get("full_name","").strip()
    ph = d.get("phone","").strip()
    ci = d.get("city","").strip()
    db = get_db()
    db.execute("UPDATE users SET full_name=?,phone=?,city=? WHERE id=?",(fn,ph,ci,session["user_id"]))
    db.commit()
    session["full_name"] = fn or session["username"]
    return jsonify({"success":True})

@app.route("/api/change-password", methods=["POST"])
@login_required
def api_change_password():
    d   = request.get_json()
    old = d.get("old_password","")
    new = d.get("new_password","")
    if len(new) < 6: return jsonify({"error":"New password must be at least 6 characters"}), 400
    db  = get_db()
    row = db.execute("SELECT password FROM users WHERE id=?",(session["user_id"],)).fetchone()
    if row["password"] != hash_password(old):
        return jsonify({"error":"Current password is incorrect"}), 401
    db.execute("UPDATE users SET password=? WHERE id=?",(hash_password(new),session["user_id"]))
    db.commit()
    return jsonify({"success":True})

@app.route("/about")
@login_required
def about_page():
    return render_template("about.html", model_ready=model_ready,
        username=session["username"], full_name=session["full_name"], role=session["role"])

# ── API: Weather ───────────────────────────────────────────────
@app.route("/api/weather")
@login_required
def weather():
    lat  = request.args.get("lat", type=float)
    lon  = request.args.get("lon", type=float)
    city = request.args.get("city","Your Location")
    if lat is None or lon is None: return jsonify({"error":"lat and lon required"}), 400
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,"
               f"precipitation,weather_code,apparent_temperature"
               f"&hourly=precipitation_probability&forecast_days=1&wind_speed_unit=kmh")
        resp = requests.get(url, timeout=8); resp.raise_for_status()
        data = resp.json(); cur = data["current"]
        rain_probs    = data.get("hourly",{}).get("precipitation_probability",[])
        avg_rain_prob = round(sum(rain_probs[:6])/max(len(rain_probs[:6]),1),1) if rain_probs else 0
        adv = get_pesticide_advisory(cur["temperature_2m"],cur["relative_humidity_2m"],
                                     cur["wind_speed_10m"],cur["precipitation"],cur["weather_code"])
        return jsonify({"success":True,"city":city,"lat":lat,"lon":lon,
            "temperature":cur["temperature_2m"],"feels_like":cur["apparent_temperature"],
            "humidity":cur["relative_humidity_2m"],"wind_kmh":cur["wind_speed_10m"],
            "rain_mm":cur["precipitation"],"rain_prob_6h":avg_rain_prob,
            "weather_code":cur["weather_code"],"weather_desc":wmo_desc(cur["weather_code"]),
            "advisory":adv})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

# ── API: Predict ───────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if not model_ready: return jsonify({"error":"Model not loaded. Run train_model.py first."}), 503
    if "file" not in request.files: return jsonify({"error":"No file uploaded."}), 400
    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename): return jsonify({"error":"Invalid file."}), 400
    try:
        pil_img  = Image.open(io.BytesIO(file.read()))
        filename = f"{uuid.uuid4().hex}.jpg"
        pil_img.convert("RGB").save(os.path.join(UPLOAD_FOLDER, filename), "JPEG", quality=85)
        result   = predict_image(pil_img)
        img_url  = f"/static/uploads/{filename}"
        db = get_db()
        if not result["is_plant"]:
            db.execute("INSERT INTO scans (user_id,image_url,plant,disease,severity,confidence,is_plant) VALUES (?,?,?,?,?,?,0)",
                       (session["user_id"],img_url,"—","Not a plant leaf","none",round(result["confidence"]*100,2)))
            db.commit()
            return jsonify({"success":True,"is_plant":False,"image_url":img_url,
                "rejection_reason":result["rejection_reason"],"rejection_stage":result["stage"],
                "top_label":result["top_label"],"plant_score":round(result["plant_score"]*100,2),
                "color_ratio":round(result.get("color_ratio",0)*100,2),
                "confidence":round(result["confidence"]*100,2),
                "top5":[{"name":n.replace("___"," → ").replace("_"," "),"confidence":round(c*100,2)} for n,c in result["top5"]]})
        from disease_info import get_disease_info
        info = get_disease_info(result["class_name"])
        db.execute("""INSERT INTO scans (user_id,image_url,plant,disease,severity,
                      confidence,remedy,is_plant) VALUES (?,?,?,?,?,?,?,1)""",
                   (session["user_id"],img_url,info["plant"],info["disease"],
                    info["severity"],round(result["confidence"]*100,2),info["remedy"]))
        db.commit()
        return jsonify({"success":True,"is_plant":True,"image_url":img_url,
            "class_name":result["class_name"],"confidence":round(result["confidence"]*100,2),
            "plant_score":round(result["plant_score"]*100,2),
            "plant":info["plant"],"disease":info["disease"],
            "severity":info["severity"],"description":info["description"],"symptoms":info["symptoms"],
            "remedy":info["remedy"],"prevention":info["prevention"],
            "top5":[{"name":n.replace("___"," → ").replace("_"," "),"confidence":round(c*100,2)} for n,c in result["top5"]]})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error":f"Prediction failed: {str(e)}"}), 500

# ── API: History ───────────────────────────────────────────────
@app.route("/api/history")
@login_required
def api_history():
    db   = get_db()
    rows = db.execute("SELECT * FROM scans WHERE user_id=? ORDER BY scanned_at DESC LIMIT 50",(session["user_id"],)).fetchall()
    return jsonify({"success":True,"scans":[dict(r) for r in rows]})

@app.route("/api/history/<int:scan_id>", methods=["DELETE"])
@login_required
def delete_scan(scan_id):
    db = get_db()
    db.execute("DELETE FROM scans WHERE id=? AND user_id=?",(scan_id,session["user_id"]))
    db.commit()
    return jsonify({"success":True})

# ── API: Tips (admin) ──────────────────────────────────────────
@app.route("/api/tips", methods=["POST"])
@admin_required
def api_add_tip():
    d = request.get_json()
    db = get_db()
    db.execute("INSERT INTO tips (title,content,category,created_by) VALUES (?,?,?,?)",
               (d.get("title",""),d.get("content",""),d.get("category","general"),session["user_id"]))
    db.commit()
    return jsonify({"success":True})

@app.route("/api/tips/<int:tid>", methods=["DELETE"])
@admin_required
def api_delete_tip(tid):
    db = get_db()
    db.execute("DELETE FROM tips WHERE id=?",(tid,))
    db.commit()
    return jsonify({"success":True})

# ── ADMIN ROUTES ───────────────────────────────────────────────
@app.route("/admin")
@admin_required
def admin_page():
    return render_template("admin.html",
        username=session["username"], full_name=session["full_name"])

@app.route("/api/admin/stats")
@admin_required
def admin_stats():
    db = get_db()
    farmers  = db.execute("SELECT COUNT(*) FROM users WHERE role='farmer'").fetchone()[0]
    scans    = db.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    admins   = db.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
    diseases = db.execute("SELECT COUNT(*) FROM scans WHERE is_plant=1").fetchone()[0]
    recent   = db.execute("""SELECT s.*,u.username,u.full_name FROM scans s
                              JOIN users u ON s.user_id=u.id
                              ORDER BY s.scanned_at DESC LIMIT 10""").fetchall()
    return jsonify({"success":True,"total_users":farmers,"total_admins":admins,
                    "total_scans":scans,"disease_scans":diseases,
                    "recent_scans":[dict(r) for r in recent]})

@app.route("/api/admin/users")
@admin_required
def admin_users():
    db   = get_db()
    rows = db.execute("""SELECT u.*,COUNT(s.id) as scan_count FROM users u
                         LEFT JOIN scans s ON u.id=s.user_id
                         GROUP BY u.id ORDER BY u.created DESC""").fetchall()
    return jsonify({"success":True,"users":[dict(r) for r in rows]})

@app.route("/api/admin/users/<int:uid>", methods=["PATCH"])
@admin_required
def admin_update_user(uid):
    d = request.get_json(); db = get_db()
    # Prevent admin from deactivating themselves
    if "active" in d and int(d["active"]) == 0 and uid == session["user_id"]:
        return jsonify({"error":"Cannot deactivate your own account"}), 400
    # Prevent admin from changing their own role
    if "role" in d and uid == session["user_id"]:
        return jsonify({"error":"Cannot change your own role"}), 400
    if "active" in d: db.execute("UPDATE users SET active=? WHERE id=?",(int(d["active"]),uid))
    if "role"   in d and d["role"] in ("farmer","admin"):
        db.execute("UPDATE users SET role=? WHERE id=?",(d["role"],uid))
    db.commit()
    return jsonify({"success":True})

@app.route("/api/admin/users/<int:uid>", methods=["DELETE"])
@admin_required
def admin_delete_user(uid):
    if uid == session["user_id"]: return jsonify({"error":"Cannot delete yourself"}), 400
    db = get_db()
    db.execute("DELETE FROM scans WHERE user_id=?",(uid,))
    db.execute("DELETE FROM users WHERE id=?",(uid,))
    db.commit()
    return jsonify({"success":True})

@app.route("/api/admin/users/<int:uid>/scans")
@admin_required
def admin_user_scans(uid):
    db   = get_db()
    rows = db.execute("SELECT * FROM scans WHERE user_id=? ORDER BY scanned_at DESC",(uid,)).fetchall()
    return jsonify({"success":True,"scans":[dict(r) for r in rows]})

@app.route("/api/admin/tips")
@admin_required
def admin_get_tips():
    db   = get_db()
    rows = db.execute("SELECT * FROM tips ORDER BY created DESC").fetchall()
    return jsonify({"success":True,"tips":[dict(r) for r in rows]})

@app.route("/health")
def health():
    return jsonify({"status":"ok","model_loaded":model_ready,
                    "gatekeeper":gatekeeper is not None,"classes":len(class_names),
                    "thresholds":{"color_ratio":COLOR_PLANT_RATIO,
                                  "plant_score":PLANT_SCORE_MIN,
                                  "conf":CONF_THRESHOLD}})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)