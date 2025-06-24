import os
import json
import joblib
from flask import Flask, jsonify, request, abort
from flask_cors import CORS
import pandas as pd
import numpy as np

# ---------- Đường dẫn artefact ----------
ART_DIR = os.path.dirname(__file__)  # cùng thư mục với serverai.py

# ---------- Nạp artefact ----------
OPTIONS = joblib.load(os.path.join(ART_DIR, "form_options.pkl"))
MODEL = joblib.load(os.path.join(ART_DIR, "xgboost_best_model.pkl"))
LABEL_ENC = joblib.load(os.path.join(ART_DIR, "label_encoder.pkl"))
TARGET_ENC = joblib.load(os.path.join(ART_DIR, "target_encoder.pkl"))
SCALER = joblib.load(os.path.join(ART_DIR, "scaler.pkl"))
FEATURE_ORDER = joblib.load(os.path.join(ART_DIR, "feature_order.pkl"))

NUMERICAL_COLS = SCALER.feature_names_in_.tolist()
BINARY_COLS = OPTIONS["symptoms"]  # danh sách các cờ Yes/No

# ---------- Flask ----------
app = Flask(__name__)
CORS(app, supports_credentials=True)

# ---------- Endpoints ----------
@app.get("/api/form-options")
def get_form_options():
    """Trả về toàn bộ option cho form React"""
    return jsonify(OPTIONS)


@app.get("/api/breeds")
def get_breeds():
    """Lấy danh sách giống theo animalType"""
    animal_type = request.args.get("animalType")
    if not animal_type:
        abort(400, "animalType query param required")
    return jsonify(OPTIONS["breeds"].get(animal_type, []))


# ---------- Tiền xử lý cho /predict ----------
def preprocess(sample: dict) -> pd.DataFrame:
    """Chuyển JSON đầu vào thành dataframe đúng format model"""
    d = sample.copy()

    # Làm sạch & ép kiểu nhiệt độ
    temp_val = d.get("Body_Temperature_C", "")
    if isinstance(temp_val, str):
        temp_val = temp_val.replace("°C", "")
    d["Body_Temperature_C"] = float(temp_val)

    # Đảm bảo Heart_Rate_BPM là float
    d["Heart_Rate_BPM"] = float(d.get("Heart_Rate_BPM", 0))

    # Tự động tính Duration_Category từ Duration_Days
    days = int(d.get("Duration_Days", 0))
    if days > 0:
        if days <= 7:
            d["Duration_Category"] = "Short"
        elif days <= 20:
            d["Duration_Category"] = "Medium"
        else:
            d["Duration_Category"] = "Long"
    else:
        d["Duration_Category"] = "Unknown"

    # Yes/No -> 1/0 cho các triệu chứng
    for col in BINARY_COLS:
        d[col] = 1 if str(d.get(col, 0)).lower() in ("yes", "1", "true") else 0

    # Đặc trưng mới
    d["Temp_HR_Ratio"] = d["Body_Temperature_C"] / (d["Heart_Rate_BPM"] + 1)
    d["Weight_Age_Ratio"] = d["Weight_kg"] / (d["Age_Years"] + 1)

    df = pd.DataFrame([d])
    df = TARGET_ENC.transform(df)
    df[NUMERICAL_COLS] = SCALER.transform(df[NUMERICAL_COLS])
    df = df.reindex(columns=FEATURE_ORDER)
    return df



@app.post("/predict")
def predict():
    if not request.is_json:
        abort(400, "JSON body required")
    sample = request.get_json()

    # --- DEBUG: in ra payload nhận được ---
    app.logger.info("Received payload:\n%s", json.dumps(sample, ensure_ascii=False, indent=2))

    try:
        df = preprocess(sample)
        probs = MODEL.predict_proba(df)[0]
        idx = int(np.argmax(probs))
        return jsonify({
            "received": sample,                 # trả về kèm payload để client xem (tùy ý)
            "predicted_disease": LABEL_ENC.inverse_transform([idx])[0],
            "probabilities": {
                cls: round(float(p), 4) for cls, p in zip(LABEL_ENC.classes_, probs)
            }
        })
    except Exception as e:
        abort(500, f"Prediction error: {e}")({
            "predicted_disease": LABEL_ENC.inverse_transform([idx])[0],
            "probabilities": {
                cls: round(float(p), 4) for cls, p in zip(LABEL_ENC.classes_, probs)
            }
        })
    except Exception as e:
        abort(500, f"Prediction error: {e}")


# ---------- Main ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
