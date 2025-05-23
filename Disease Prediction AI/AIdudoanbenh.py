import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from category_encoders import TargetEncoder
from xgboost import XGBClassifier

# 1. Đọc dữ liệu
df = pd.read_csv("synthetic_pet_disease_data_extended_1.csv", encoding="ISO-8859-1")

# 2. Làm sạch nhiệt độ cơ thể
df['Body_Temperature_C'] = df['Body_Temperature_C'].astype(str).str.replace('°C', '', regex=False).astype(float)

# 3. Chuyển các cột Yes/No thành 1/0
binary_cols = [
    'Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing',
    'Labored_Breathing', 'Lameness', 'Skin_Lesions',
    'Nasal_Discharge', 'Eye_Discharge',
    'Weight_Loss', 'Fever', 'Lethargy'
]

for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# 4. Tạo đặc trưng mới
df['Temp_HR_Ratio'] = df['Body_Temperature_C'] / (df['Heart_Rate_BPM'] + 1)
df['Weight_Age_Ratio'] = df['Weight_kg'] / (df['Age_Years'] + 1)

# 5. Đặc trưng và nhãn
categorical_cols = ['Animal_Type', 'Breed', 'Gender', 'Duration_Category', 'Severity', 'Season', 'Living_Area']
numerical_cols = ['Age_Years', 'Weight_kg', 'Duration_Days', 'Body_Temperature_C', 'Heart_Rate_BPM',
                  'Temp_HR_Ratio', 'Weight_Age_Ratio'] + binary_cols

X = df[categorical_cols + numerical_cols]
y = df['Disease']

# 6. Mã hóa nhãn
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 7. Tách dữ liệu
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 8. Mã hóa phân loại bằng TargetEncoder
te = TargetEncoder(cols=categorical_cols)
X_train = te.fit_transform(X_train_raw, y_train)
X_test = te.transform(X_test_raw)

# 9. Chuẩn hóa dữ liệu số
scaler = RobustScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# 10. Cân bằng dữ liệu bằng SMOTE
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# 11. Khởi tạo mô hình XGBoost
xgb_model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(le.classes_),
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_jobs=-1,
    random_state=42
)

# 12. Tối ưu tham số với RandomizedSearchCV
param_dist = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 500, 1000],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'min_child_weight': [1, 3, 5]
}

search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)
best_model = search.best_estimator_
print("Best Params:", search.best_params_)

# 13. Dự đoán và đánh giá
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# Lưu mô hình
joblib.dump(best_model, 'xgboost_best_model.pkl')
# Lưu encoder
joblib.dump(le, 'label_encoder.pkl')

# 14.
# Huấn luyện lại best_model để lấy logloss trong quá trình huấn luyện
eval_set = [(X_train, y_train), (X_test, y_test)]
# Khởi tạo model với eval_metric trong constructor
best_model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(le.classes_),
    use_label_encoder=False,
    n_jobs=-1,
    random_state=42,
    eval_metric=["mlogloss", "merror"]
)

# Huấn luyện model với eval_set, KHÔNG dùng callbacks
best_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# Lấy kết quả eval
results = best_model.evals_result()
train_logloss = results['validation_0']['mlogloss']
test_logloss = results['validation_1']['mlogloss']

# Vẽ biểu đồ logloss
plt.figure()
plt.plot(train_logloss, label='Train Logloss')
plt.plot(test_logloss, label='Test Logloss')
plt.xlabel('Round')
plt.ylabel('Logloss')
plt.title('XGBoost Logloss over Iterations')
plt.legend()
plt.grid()
plt.show(block=False)

# Vẽ ma trận nhầm lẫn
fig, ax = plt.subplots()
cm = confusion_matrix(y_test, best_model.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=ax, xticks_rotation=45)
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.show()

