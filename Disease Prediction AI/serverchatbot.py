from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import traceback
from dotenv import load_dotenv
import os
import json

import spacy
import re

# Load spaCy English model 1 lần
nlp = spacy.load("en_core_web_sm")

def extract_features_from_text(text, answer=""):

    features = {
        "Animal_Type": "Unknown",
        "Breed": "Unknown",
        "Gender": "Unknown",
        "Duration_Category": "Unknown",
        "Severity": "Unknown",
        "Season": "Unknown",
        "Living_Area": "Unknown",
        "Age_Years": 0,
        "Weight_kg": 0,
        "Duration_Days": 0,
        "Body_Temperature_C": 0,
        "Heart_Rate_BPM": 0,
        "Temp_HR_Ratio": 0,
        "Weight_Age_Ratio": 0,
        "Appetite_Loss": 0,
        "Vomiting": 0,
        "Diarrhea": 0,
        "Coughing": 0,
        "Labored_Breathing": 0,
        "Lameness": 0,
        "Skin_Lesions": 0,
        "Nasal_Discharge": 0,
        "Eye_Discharge": 0,
        "Weight_Loss": 0,
        "Fever": 0,
        "Lethargy": 0
    }

    # Gộp văn bản đầu vào và câu trả lời bổ sung từ client
    # Nếu answer là dict thì chuyển thành chuỗi
    if isinstance(answer, dict):
        answer_str = " ".join([f"{k}: {v}" for k, v in answer.items()])
    else:
        answer_str = str(answer)  # phòng trường hợp answer là None hoặc str

    full_text = (text + " " + answer_str).strip().lower()
    doc = nlp(full_text)

    full_text = (text + " " + answer_str).strip().lower()

    doc = nlp(full_text)

    print("\n🔍 Đang phân tích văn bản:")
    print(full_text)

    # Animal_Type
    animals_vi = {"chó": "Dog", "mèo": "Cat", "chim": "Bird", "thỏ": "Rabbit", "ngựa": "Horse"}
    for token in doc:
        if token.text in animals_vi:
            features["Animal_Type"] = animals_vi[token.text]
            break
    print(f"🐾 Animal_Type: {features['Animal_Type']}")

    # Gender
    if any(tok.text in ["đực", "nam", "male", "he", "him"] for tok in doc):
        features["Gender"] = "Male"
    elif any(tok.text in ["cái", "nữ", "female", "she", "her"] for tok in doc):
        features["Gender"] = "Female"
    print(f"🚻 Gender: {features['Gender']}")

    # Season
    if "mùa hè" in full_text:
        features["Season"] = "Summer"
    elif "mùa đông" in full_text:
        features["Season"] = "Winter"
    elif "mùa xuân" in full_text:
        features["Season"] = "Spring"
    elif "mùa thu" in full_text:
        features["Season"] = "Autumn"
    print(f"🌦️ Season: {features['Season']}")

    # Living_Area
    if "thành thị" in full_text or "đô thị" in full_text:
        features["Living_Area"] = "Urban"
    elif "nông thôn" in full_text:
        features["Living_Area"] = "Rural"
    print(f"🏙️ Living_Area: {features['Living_Area']}")

    # Severity (theo mô tả mức độ)
    if "triệu chứng nhẹ" in full_text:
        features["Severity"] = "Mild"
    elif "triệu chứng trung bình" in full_text:
        features["Severity"] = "Moderate"
    elif "triệu chứng nặng" in full_text:
        features["Severity"] = "Severe"

    # Age_Years
    age_match = re.search(r"(\d+)\s*tuổi|tuổi\s*(\d+)", full_text)
    if age_match:
        age = age_match.group(1) or age_match.group(2)
        if age:
            features["Age_Years"] = float(age)
            print(f"🎂 Age_Years: {features['Age_Years']}")

    # Body_Temperature_C
    temp_match = re.search(r"(\d{1,3}\.?\d*)\s*(độ|°)\s*c", full_text)
    if temp_match:
        features["Body_Temperature_C"] = float(temp_match.group(1))
        print(f"🌡️ Body_Temperature_C: {features['Body_Temperature_C']}")

    # Heart_Rate_BPM
    hr_match = re.search(r"(\d{2,3})\s*(nhịp|lần)?\s*(\/)?\s*(phút|bpm|beats per minute)", full_text)
    if hr_match:
        features["Heart_Rate_BPM"] = float(hr_match.group(1))
        print(f"❤️ Heart_Rate_BPM: {features['Heart_Rate_BPM']}")

    # Weight_kg
    weight_match = re.search(r"(\d{1,3}\.?\d*)\s*(kg|kilograms?|kgm)", full_text)
    if weight_match:
        features["Weight_kg"] = float(weight_match.group(1))
    print(f"⚖️ Weight_kg: {features['Weight_kg']}")

    # Duration_Days
    duration_match = re.search(r"(\d+)\s*(days?|ngày)", full_text)
    if duration_match:
        features["Duration_Days"] = int(duration_match.group(1))
    print(f"⏳ Duration_Days: {features['Duration_Days']}")

    # Duration_Category
    days = features["Duration_Days"]
    if days > 0:
        if days <= 7:
            features["Duration_Category"] = "Short"
        elif days <= 20:
            features["Duration_Category"] = "Medium"
        else:
            features["Duration_Category"] = "Long"
    print(f"📊 Duration_Category: {features['Duration_Category']}")

    # Breed
    breeds = {
        'Dog': ['Labrador', 'Poodle', 'Bulldog', 'Beagle', 'German Shepherd'],
        'Cat': ['Persian', 'Siamese', 'Maine Coon', 'Ragdoll', 'Bengal']
    }

    # Tạo danh sách tất cả giống, chuẩn hoá viết hoa đầu từ để dễ dò
    all_breeds = []
    for kind in breeds.values():
        all_breeds.extend(kind)

    # Tạo regex pattern để tìm các giống trong đoạn text
    # Lưu ý: sắp xếp theo độ dài giảm dần để tránh nhầm lẫn "German Shepherd" bị trùng với "Shepherd"
    all_breeds_sorted = sorted(all_breeds, key=lambda x: len(x.split()), reverse=True)
    # Ví dụ: ['German Shepherd', 'Maine Coon', 'Labrador', ...]

    # Ghép pattern regex để tìm các giống (ignore case)
    pattern = r'\b(' + '|'.join(re.escape(breed) for breed in all_breeds_sorted) + r')\b'

    matches = re.findall(pattern, full_text, re.IGNORECASE)

    if matches:
        # Lấy giống đầu tiên tìm được (hoặc bạn có thể lưu hết nếu muốn)
        breed_found = matches[0]
        # Chuẩn hoá viết hoa đầu chữ
        breed_found = ' '.join(word.capitalize() for word in breed_found.split())
        features["Breed"] = breed_found
    else:
        features["Breed"] = None

    # Symptoms và các từ khóa
    symptom_keywords = {
        "Appetite_Loss": ["appetite_loss", "appetite loss", "loss of appetite", "chán ăn", "bỏ ăn"],
        "Vomiting": ["vomiting", "vomit", "nôn"],
        "Diarrhea": ["diarrhea", "tiêu chảy"],
        "Coughing": ["cough", "coughing", "ho"],
        "Labored_Breathing": ["labored_breathing","labored breathing", "difficulty breathing", "khó thở"],
        "Lameness": ["lameness", "limping", "đi khập khiễng"],
        "Skin_Lesions": ["skin_lesions", "skin lesions", "lesions on skin", "tổn thương da"],
        "Nasal_Discharge": ["nasal_discharge", "nasal discharge", "runny nose", "chảy nước mũi"],
        "Eye_Discharge": ["eye_discharge", "eye discharge", "watery eyes", "chảy nước mắt"],
        "Weight_Loss": ["weight_loss", "weight loss", "losing weight", "giảm cân", "sút cân"],
        "Fever": ["fever", "high temperature", "sốt"],
        "Lethargy": ["lethargy", "tired", "low energy", "mệt mỏi"]
    }

    for key, phrases in symptom_keywords.items():
        features[key] = 0  # khởi đầu là 0
        for phrase in phrases:
            # Tạo pattern regex tìm chính xác từ khóa (có giới hạn từ \b), không phân biệt hoa thường
            pattern = r"\b" + re.escape(phrase) + r"\b"
            for match in re.finditer(pattern, full_text, re.IGNORECASE):
                start_idx = match.start()
                # Lấy tối đa 10 ký tự trước để kiểm tra phủ định "không" (đã cách và có thể có dấu câu)
                prefix_start = max(0, start_idx - 10)
                prefix_text = full_text[prefix_start:start_idx].lower()
                
                # Kiểm tra xem có phủ định "không" ngay trước từ khóa không
                # Phủ định "không" phải đứng riêng biệt, cách 0-3 ký tự trước (có thể có dấu cách, dấu câu)
                # Ví dụ: "không nôn", "không bị nôn"
                # Đơn giản: kiểm tra xem có "không" trong khoảng prefix gần trước và không kèm theo từ khác
                # Dùng regex để tìm phủ định gần từ khóa
                negation_pattern = r"không\s{0,3}$"
                if not re.search(negation_pattern, prefix_text):
                    features[key] = 1
                    break
            if features[key] == 1:
                break
        print(f"🩺 {key}: {features[key]}")

    # Technical features
    if features["Heart_Rate_BPM"] > 0:
        features["Temp_HR_Ratio"] = features["Body_Temperature_C"] / (features["Heart_Rate_BPM"] + 1)
    if features["Age_Years"] > 0:
        features["Weight_Age_Ratio"] = features["Weight_kg"] / (features["Age_Years"] + 1)

    print(f"📈 Temp_HR_Ratio: {features['Temp_HR_Ratio']:.4f}")
    print(f"📉 Weight_Age_Ratio: {features['Weight_Age_Ratio']:.4f}")

    print("\n🧠 TỔNG HỢP FEATURES TRÍCH XUẤT:")
    print(json.dumps(features, indent=2, ensure_ascii=False))

    return features, None, full_text


print("🚀 Đang tải model và các encoder...")
model = joblib.load('xgboost_best_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
target_encoder = joblib.load("target_encoder.pkl")  # Encode 7 cột categorical
scaler = joblib.load("scaler.pkl")
print("✅ Tải xong model và encoder!")

app = Flask(__name__)
CORS(app)

field_questions = {
    'Animal_Type': "🐾 Thú cưng của bạn là chó, mèo hay loại khác?",
    'Breed': "🧬 Giống của thú cưng là gì?",
    'Gender': "⚥ Thú cưng là đực hay cái?",
    'Season': "📅 Thời điểm thú cưng bị bệnh là mùa nào?",
    'Living_Area': "🏠 Thú cưng sống ở thành thị hay nông thôn?",
    'Severity': "📈 Theo bạn, tình trạng thú cưng hiện tại là triệu chứng nhẹ (vẫn hoạt động bình thường), triệu chứng trung bình (giảm hoạt động, mệt mỏi), hay triệu chứng nặng (liệt, không ăn uống, cần cấp cứu)?",

    'Age_Years': "📆 Thú cưng bao nhiêu tuổi?",
    'Weight_kg': "⚖️ Thú cưng nặng khoảng bao nhiêu kg?",
    'Duration_Days': "⏱️ Số ngày bị bệnh chính xác là bao nhiêu?",
    'Body_Temperature_C': "🌡️ Nhiệt độ cơ thể hiện tại là bao nhiêu?",
    'Heart_Rate_BPM': "❤️ Nhịp tim mỗi phút của thú cưng là bao nhiêu?",

    'Appetite_Loss': "🐶 Thú cưng có bỏ ăn không?",
    'Vomiting': "🤮 Thú cưng có bị nôn không?",
    'Diarrhea': "💩 Có bị tiêu chảy không?",
    'Coughing': "😷 Có ho không?",
    'Labored_Breathing': "😮‍💨 Có thở khó khăn không?",
    'Lameness': "🐾 Có đi khập khiễng không?",
    'Skin_Lesions': "🩹 Có vết thương, vết loét trên da không?",
    'Nasal_Discharge': "👃 Có chảy nước mũi không?",
    'Eye_Discharge': "👁️ Có chảy nước mắt không?",
    'Weight_Loss': "📉 Có bị sút cân không?",
    'Fever': "🌡️ Có bị sốt không?",
    'Lethargy': "💤 Có mệt mỏi, lờ đờ không?",
}
# Danh sách triệu chứng cần hỏi
binary_cols = [
    'Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing',
    'Labored_Breathing', 'Lameness', 'Skin_Lesions',
    'Nasal_Discharge', 'Eye_Discharge',
    'Weight_Loss', 'Fever', 'Lethargy'
]

# Hàm kiểm tra các trường còn thiếu
def find_missing_fields(features):
    missing = []

    unknown_string_fields = [
        'Animal_Type', 'Breed', 'Gender', 'Duration_Category', 'Season', 'Living_Area', 'Severity'
    ]

    zero_numeric_fields = [
        'Age_Years', 'Weight_kg', 'Duration_Days', 'Body_Temperature_C', 'Heart_Rate_BPM'
    ]

    for key in unknown_string_fields:
        if features.get(key, 'Unknown') == 'Unknown':
            missing.append(key)

    for key in zero_numeric_fields:
        if features.get(key, 0) == 0:
            missing.append(key)

    # Nếu tất cả triệu chứng đều bằng 0 hoặc không có triệu chứng nào được khai báo
    all_symptoms_zero = all(features.get(col, 0) == 0 for col in binary_cols)
    if all_symptoms_zero:
        missing.append('SYMPTOMS_UNKNOWN')  # đánh dấu cần hỏi chung triệu chứng

    return missing

@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        if request.method == "GET":
            # Lấy description và answer từ query params
            description = request.args.get("description")
            answer = request.args.get("answers")

            if not description:
                return jsonify({"error": "Missing 'description' parameter"}), 400

            features, error_text, full_description = extract_features_from_text(description, answer)

            if error_text:
                return jsonify({"error": "Không thể trích xuất dữ liệu từ mô tả tự nhiên", "detail": error_text}), 400

            return jsonify({
                "full_text": full_description
            }), 200

        # Các cột đặc trưng như khi train
        binary_cols = [
            'Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing',
            'Labored_Breathing', 'Lameness', 'Skin_Lesions',
            'Nasal_Discharge', 'Eye_Discharge',
            'Weight_Loss', 'Fever', 'Lethargy'
        ]
        categorical_cols = ['Animal_Type', 'Breed', 'Gender', 'Duration_Category', 'Severity', 'Season', 'Living_Area']
        numerical_cols = ['Age_Years', 'Weight_kg', 'Duration_Days', 'Body_Temperature_C', 'Heart_Rate_BPM',
                          'Temp_HR_Ratio', 'Weight_Age_Ratio'] + binary_cols

        print("📩 Nhận request POST mới từ client...")

        data = request.json
        print(f"📦 Dữ liệu JSON nhận được:\n{json.dumps(data, indent=2, ensure_ascii=False)}")

        description = data.get("description")
        answer = data.get("answers")

        print(f"📝 Description hiện tại: {description}")
        print(f"💬 Các câu trả lời triệu chứng bổ sung (answer): {answer}")

        if not description:
            return jsonify({"error": "Missing 'description' field"}), 400

        features, error_text, full_description = extract_features_from_text(description, answer)

        if error_text:
            print(f"⚠️ Lỗi khi trích xuất đặc trưng: {error_text}")
            return jsonify({"error": "Không thể trích xuất dữ liệu từ mô tả tự nhiên", "detail": error_text}), 400
        if not features:
            print("⚠️ Không trích xuất được đặc trưng từ mô tả.")
            return jsonify({"error": "Dữ liệu JSON không hợp lệ"}), 400

        print("✅ Trích xuất đặc trưng thành công.")
        print(f"🧾 Đặc trưng trích xuất:\n{features}")

        # Kiểm tra các trường bị thiếu (nếu có hàm find_missing_fields)
        missing_fields = find_missing_fields(features)

        if not answer and 'SYMPTOMS_UNKNOWN' in missing_fields:
            symptom_list = [
                field_questions[col] for col in binary_cols if col in field_questions
            ]
            return jsonify({
                "message": "Thú cưng có biểu hiện nào sau đây không?",
                "symptoms": symptom_list,
                "ask_symptom_confirmation": True
            }), 200

        elif missing_fields:
            non_symptom_fields = [field for field in missing_fields if field not in binary_cols]
            questions = [field_questions[field] for field in non_symptom_fields if field in field_questions]
            if questions:
                print("📋 Thiếu thông tin, cần hỏi thêm:", questions)
                return jsonify({
                    "message": "Tôi cần thêm thông tin để chẩn đoán chính xác hơn.",
                    "questions": questions
                }), 200

        df = pd.DataFrame([features])
        print("🧮 DataFrame đầu vào:\n", df)

        # Tính toán các đặc trưng kỹ thuật
        try:
            df['Temp_HR_Ratio'] = df['Body_Temperature_C'] / (df['Heart_Rate_BPM'] + 1)
            df['Weight_Age_Ratio'] = df['Weight_kg'] / (df['Age_Years'] + 1)
        except Exception as e:
            print(f"❌ Lỗi khi tính đặc trưng kỹ thuật: {e}")
            return jsonify({"error": "Lỗi khi tính toán đặc trưng kỹ thuật", "detail": str(e)}), 500

        print("🧪 Đã tính toán các đặc trưng kỹ thuật.")

        expected_cols = set(categorical_cols + numerical_cols)
        missing_cols = expected_cols - set(df.columns)
        if missing_cols:
            print(f"❌ Thiếu cột đầu vào: {missing_cols}")
            return jsonify({'error': f'Missing columns: {missing_cols}'}), 400

        print("✅ Đầy đủ các cột đầu vào.")

        print("🔄 Đang mã hóa các cột phân loại...")
        try:
            df_encoded = target_encoder.transform(df)
            df_cat_encoded = df_encoded[categorical_cols]
            print("✅ Mã hóa cột phân loại thành công.")
        except Exception as e:
            print(f"❌ Lỗi mã hóa cột phân loại: {e}")
            return jsonify({"error": "Lỗi mã hóa cột phân loại", "detail": str(e)}), 500

        print("📏 Đang chuẩn hóa các cột số...")
        try:
            df_num_scaled = pd.DataFrame(scaler.transform(df[numerical_cols]), columns=numerical_cols)
            print("✅ Chuẩn hóa cột số thành công.")
        except Exception as e:
            print(f"❌ Lỗi chuẩn hóa cột số: {e}")
            return jsonify({"error": "Lỗi chuẩn hóa cột số", "detail": str(e)}), 500

        X_pred = pd.concat([df_cat_encoded.reset_index(drop=True), df_num_scaled.reset_index(drop=True)], axis=1)
        print(f"🧮 Kích thước dữ liệu đầu vào cho model: {X_pred.shape}")
        print(f"📋 Tên các cột đầu vào model:\n{X_pred.columns.tolist()}")

        expected_feature_count = len(categorical_cols) + len(numerical_cols)
        if X_pred.shape[1] != expected_feature_count:
            print(f"❌ Sai số lượng cột: Expected {expected_feature_count}, got {X_pred.shape[1]}")
            return jsonify({"error": f"Sai số lượng cột đầu vào. Expected {expected_feature_count}, got {X_pred.shape[1]}"}), 400

        print("🔮 Đang dự đoán...")
        prediction = model.predict(X_pred)
        proba = model.predict_proba(X_pred)

        print(f"📊 Label encoder classes: {label_encoder.classes_}")
        print(f"📈 Prediction numeric label: {prediction}")

        try:
            result = label_encoder.inverse_transform(prediction)
        except Exception as e:
            print(f"❌ Lỗi khi giải mã nhãn dự đoán: {e}")
            return jsonify({"error": "Lỗi khi giải mã nhãn dự đoán", "detail": str(e)}), 500

        confidence_score = proba[0].max()

        print(f"🎯 Kết quả dự đoán: {result[0]} với tỉ lệ chính xác: {confidence_score:.4f}")

        return jsonify({
            'prediction': result[0],
            'confidence': float(confidence_score),
            'full_text': full_description
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    print("🟢 Server Flask đang chạy tại http://localhost:8000")
    app.run(debug=True, port=8000)
