from flask import Flask, render_template, request, jsonify, send_from_directory
import sqlite3, requests, threading, time, os
from datetime import datetime
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import xgboost as xgb
import json
from googletrans import Translator
translator = Translator()



from gtts import gTTS
LAST_ENGLISH_TEXT = ""


app = Flask(__name__)

# ================= CONFIG =================
DB_NAME = "crop.db"
API_URL = "https://aislyntech.com/Api/oo-get.php"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
LAST_SPEECH_TEXT = "Prediction not done yet"
TTS_LANG = {
    "en": "en",
    "ta": "ta",
    "kn": "kn",
    "te": "te",
    "hi": "hi",
    "ml": "ml",
    "mr": "mr"
}

LAST_ENGLISH_TEXT = ""
LAST_DISPLAY_TEXT = ""


# ================= LOAD MODELS =================
image_model = load_model("hybrid_cnn_vgg16_final.keras",compile=False)

risk_model = xgb.XGBClassifier()
risk_model.load_model("xgb_risk_model.json")

RISK_LABELS = ["Low", "Medium", "High"]

CLASS_NAMES = [
    "Bacterial spot",
    "Black rot",
    "Common rust",
    "Powdery mildew",
    "Septoria leaf spot",
    "Normal"
]

live_running = False
live_thread = None



# ================= RECOMMENDATIONS =================
DISEASE_RECOMMENDATIONS = {
    "Bacterial spot": {
        "cause": "Bacterial infection due to warm and humid conditions",
        "prevention": [
            "Use certified disease-free seeds",
            "Avoid overhead irrigation",
            "Ensure proper field drainage"
        ],
        "treatment": [
            "Apply copper-based bactericides",
            "Remove infected plant debris",
            "Practice crop rotation"
        ]
    },

    "Black rot": {
        "cause": "Fungal infection caused by Xanthomonas bacteria",
        "prevention": [
            "Avoid planting in infected soil",
            "Maintain proper spacing between plants",
            "Use resistant crop varieties"
        ],
        "treatment": [
            "Apply appropriate fungicides",
            "Remove and destroy infected plants",
            "Avoid working in wet fields"
        ]
    },

    "Common rust": {
        "cause": "Fungal spores spread through wind",
        "prevention": [
            "Plant rust-resistant varieties",
            "Monitor crops regularly",
            "Ensure balanced fertilization"
        ],
        "treatment": [
            "Apply sulfur or triazole fungicides",
            "Remove affected leaves",
            "Improve air circulation"
        ]
    },

    "Powdery mildew": {
        "cause": "Fungal growth favored by dry days and cool nights",
        "prevention": [
            "Avoid excess nitrogen fertilizers",
            "Ensure good sunlight exposure",
            "Maintain plant spacing"
        ],
        "treatment": [
            "Apply neem oil or sulfur spray",
            "Use potassium bicarbonate solutions",
            "Remove infected leaves"
        ]
    },

    "Septoria leaf spot": {
        "cause": "Fungal disease due to prolonged leaf wetness",
        "prevention": [
            "Avoid wetting leaves during irrigation",
            "Use mulch to reduce soil splash",
            "Rotate crops yearly"
        ],
        "treatment": [
            "Apply chlorothalonil fungicide",
            "Remove infected leaves",
            "Destroy crop residues"
        ]
    },

    "Normal": {
        "cause": "Healthy crop condition",
        "prevention": [
            "Continue regular monitoring",
            "Maintain balanced irrigation",
            "Use recommended fertilizers"
        ],
        "treatment": [
            "No treatment required",
            "Maintain good agricultural practices"
        ]
    }
}

def temperature_level(t):
    if t < 20:
        return "Low"
    elif t <= 35:
        return "Medium"
    else:
        return "High"

def humidity_level(h):
    if h < 40:
        return "Low"
    elif h <= 70:
        return "Medium"
    else:
        return "High"

def moisture_level(m):
    if m < 30:
        return "Low"
    elif m <= 60:
        return "Medium"
    else:
        return "High"

def light_level(l):
    return "Light" if int(l) == 0 else "Dark"
def risk(temp, humidity, moisture, light=None):

    t_level = temperature_level(temp)
    h_level = humidity_level(humidity)
    m_level = moisture_level(moisture)

    levels = [t_level, h_level, m_level]

    high_count = levels.count("High")
    medium_count = levels.count("Medium")

    # HIGH RISK
    if high_count >= 2:
        return "High", 85

    # MEDIUM RISK
    if high_count == 1 or medium_count >= 2:
        return "Medium", 60

    # LOW RISK
    return "Low", 25


# ================= DATABASE =================
def get_db():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db = get_db()

    db.execute("""
    CREATE TABLE IF NOT EXISTS sensor_data(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        temperature REAL,
        humidity REAL,
        moisture REAL,
        light REAL,
        latitude REAL,
        longitude REAL,
        time TEXT
    )
    """)

    db.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT,
        disease TEXT,
        confidence REAL,
        risk_category TEXT,
        recommendation_text TEXT,
        risk_percentage REAL,
        temperature REAL,
        humidity REAL,
        moisture REAL,
        light REAL,
        latitude REAL,
        longitude REAL,
        time TEXT
    )
    """)
    db.commit()

init_db()


# ================= LIVE SENSOR =================
def live_loop():
    global live_running
    while live_running:
        try:
            r = requests.get(API_URL, timeout=5).json()["data"][0]
            
            print("API RAW RESPONSE:", r)

            db = get_db()
            db.execute("""
            INSERT INTO sensor_data(
                temperature, humidity, moisture, light,
                latitude, longitude, time
            ) VALUES (?,?,?,?,?,?,?)
            """, (
                float(r["temperature"]),
                float(r["humidity"]),
                float(r["moisture"]),
                float(r["light"]),
                float(r["latitude"]),
                float(r["longitude"]),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            db.commit()

        except Exception as e:
            print("Sensor error:", e)

        time.sleep(3)

# ================= IMAGE PREDICTION =================
def predict_disease(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    pred = image_model.predict(img, verbose=0)
    return CLASS_NAMES[np.argmax(pred)], float(np.max(pred) * 100)

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start_live")
def start_live():
    global live_running, live_thread
    if not live_running:
        live_running = True
        live_thread = threading.Thread(target=live_loop, daemon=True)
        live_thread.start()
    return jsonify({"status": "started"})

@app.route("/stop_live")
def stop_live():
    global live_running
    live_running = False
    return jsonify({"status": "stopped"})

@app.route("/live_data")
def live_data():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM sensor_data ORDER BY id DESC LIMIT 10"
    ).fetchall()

    data = []
    for r in rows:
        temp = r["temperature"]
        hum = r["humidity"]
        mois = r["moisture"]
        light = r["light"]

        data.append({
            "temperature": temp,
            "temperature_level": temperature_level(temp),

            "humidity": hum,
            "humidity_level": humidity_level(hum),

            "moisture": mois,
            "moisture_level": moisture_level(mois),

            "light": light,
            "light_status": light_level(light),

            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "time": r["time"]
        })

    return jsonify(data)


@app.route("/text_to_speech")
def text_to_speech():
    lang = request.args.get("lang", "en")

    filename = f"response_{int(time.time())}.mp3"

    tts = gTTS(
        text=LAST_SPEECH_TEXT,
        lang=TTS_LANG.get(lang, "en")
    )
    tts.save(filename)

    return send_from_directory(".", filename, mimetype="audio/mpeg")
@app.route("/debug_db")
def debug_db():
    db = get_db()
    s_count = db.execute("SELECT COUNT(*) FROM sensor_data").fetchone()[0]
    p_count = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    return jsonify({
        "sensor_rows": s_count,
        "prediction_rows": p_count
    })


import speech_recognition as sr

@app.route("/speech_to_text")
def speech_to_text():
    lang = request.args.get("lang", "en")

    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source, timeout=5)

    try:
        text = r.recognize_google(audio, language=lang)
    except:
        text = "Could not recognize speech"

    return jsonify({"text": text})

@app.route("/predict", methods=["POST"])
def predict():
    lang = request.form.get("lang", "en")
    image = request.files["image"]
    path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(path)

    db = get_db()
    s = db.execute(
        "SELECT * FROM sensor_data ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not s:
        return jsonify({"error": "Start Live Sensor First"})

    temperature = s["temperature"]
    humidity = s["humidity"]
    moisture = s["moisture"]
    light = s["light"]

    temp_level = temperature_level(temperature)
    humidity_level_val = humidity_level(humidity)
    moisture_level_val = moisture_level(moisture)
    light_status = light_level(light)

    

    disease, confidence = predict_disease(path)
    recommendation = DISEASE_RECOMMENDATIONS.get(disease, {})
    recommendation_text = json.dumps(recommendation)


    risk_category, danger_risk = risk(
    s["temperature"],
    s["humidity"],
    s["moisture"],
    s["light"]
)

    alert = ""
    if danger_risk >= 70:
        alert = "HIGH RISK! Immediate action required"

    # STORE ALL VALUES USED FOR PREDICTION
    db.execute("""
    INSERT INTO predictions(
        image_name, disease, confidence,
        risk_category, recommendation_text,
        risk_percentage,
        temperature, humidity, moisture, light,
        latitude, longitude, time
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        image.filename,
        disease,
        round(confidence, 2),
        risk_category,
        recommendation_text,
        danger_risk,
        s["temperature"],
        s["humidity"],
        s["moisture"],
        s["light"],
        s["latitude"],
        s["longitude"],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    db.commit()
    db.close()
    global LAST_SPEECH_TEXT, LAST_ENGLISH_TEXT, LAST_DISPLAY_TEXT

    # Build recommendation text
    cause = recommendation.get("cause", "")
    prevention = recommendation.get("prevention", [])
    treatment = recommendation.get("treatment", [])

    prevention_text = ". ".join(prevention)
    treatment_text = ". ".join(treatment)

# FULL ENGLISH SPEECH CONTENT
    LAST_ENGLISH_TEXT = (
    f"Disease detected is {disease}. "
    f"Confidence level is {round(confidence, 2)} percent. "
    f"Risk category is {risk_category}. "
    f"Risk percentage is {danger_risk} percent. "
    f"Cause of the disease is: {cause}. "
    f"Prevention methods are: {prevention_text}. "
    f"Treatment methods are: {treatment_text}."
    )   

# Default = English
    LAST_DISPLAY_TEXT = LAST_ENGLISH_TEXT
    LAST_SPEECH_TEXT = LAST_ENGLISH_TEXT

# Optional high risk alert
    if danger_risk >= 70:
        LAST_SPEECH_TEXT += " Warning! High risk detected. Immediate action is required."
    return jsonify({
        "image_name": image.filename,
        "disease": disease,
        "confidence": round(confidence, 2),
        "risk_category": risk_category,
        "risk_percentage": danger_risk,
        "alert": alert,
        "recommendation": recommendation,
        "sensor": {
    "temperature": temperature,
    "temperature_level": temp_level,

    "humidity": humidity,
    "humidity_level": humidity_level_val,

    "moisture": moisture,
    "moisture_level": moisture_level_val,

    "light": light,
    "light_status": light_status,

    "latitude": s["latitude"],
    "longitude": s["longitude"]
}

    })


@app.route("/change_language")
def change_language():
    global LAST_ENGLISH_TEXT, LAST_DISPLAY_TEXT, LAST_SPEECH_TEXT

    lang = request.args.get("lang", "en")

    # No prediction yet
    if not LAST_ENGLISH_TEXT or LAST_ENGLISH_TEXT.strip() == "":
        return jsonify({"text": ""})

    # English → no translation
    if lang == "en":
        LAST_DISPLAY_TEXT = LAST_ENGLISH_TEXT
        LAST_SPEECH_TEXT = LAST_ENGLISH_TEXT
        return jsonify({"text": LAST_DISPLAY_TEXT})

    # Translate safely
    try:
        translated = translator.translate(
            LAST_ENGLISH_TEXT,
            dest=lang
        ).text

        LAST_DISPLAY_TEXT = translated
        LAST_SPEECH_TEXT = translated

        return jsonify({"text": translated})

    except Exception as e:
        print("Translation Error:", e)

        # Fallback to English (VERY IMPORTANT)
        LAST_DISPLAY_TEXT = LAST_ENGLISH_TEXT
        LAST_SPEECH_TEXT = LAST_ENGLISH_TEXT

        return jsonify({
            "text": LAST_DISPLAY_TEXT,
            "error": "Translation failed, showing English"
        })


@app.route("/db_data")
def db_data():
    db = get_db()
    s = db.execute("SELECT * FROM sensor_data ORDER BY id DESC").fetchall()
    p = db.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    return jsonify({
        "sensor": [dict(x) for x in s],
        "predictions": [dict(x) for x in p]
    })

@app.route("/delete_db")
def delete_db():
    db = get_db()
    db.execute("DELETE FROM sensor_data")
    db.execute("DELETE FROM predictions")
    db.commit()
    return jsonify({"status": "database cleared"})

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
@app.route("/stop_voice")
def stop_voice():
    return jsonify({"status": "stopped"})


if __name__ == "__main__":
    app.run(debug=True)



