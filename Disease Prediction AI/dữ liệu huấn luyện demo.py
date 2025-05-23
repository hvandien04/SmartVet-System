import pandas as pd
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# Các loại động vật và giống
animal_types = ['Dog', 'Cat']
breeds = {
    'Dog': ['Labrador', 'Poodle', 'Bulldog', 'Beagle', 'German Shepherd'],
    'Cat': ['Persian', 'Siamese', 'Maine Coon', 'Ragdoll', 'Bengal']
}
genders = ['Male', 'Female']

# Bệnh theo từng loại động vật
diseases = {
    'Dog': [
        'Parvovirus', 'Canine Distemper', 'Leptospirosis',
        'Rabies', 'Kennel Cough', 'Lyme Disease', 'Heartworm'
    ],
    'Cat': [
        'FeLV', 'FIV', 'Diabetes',
        'Rabies', 'Panleukopenia', 'Upper Respiratory Infection', 'Hyperthyroidism'
    ]
}


# Mức độ bệnh (Severity)
severity_levels = ['Mild', 'Moderate', 'Severe']
severity_probs = {
    'Parvovirus': [0.5, 0.3, 0.2],
    'Canine Distemper': [0.4, 0.4, 0.2],
    'Leptospirosis': [0.6, 0.3, 0.1],
    'FeLV': [0.5, 0.4, 0.1],
    'FIV': [0.7, 0.2, 0.1],
    'Diabetes': [0.6, 0.3, 0.1],
    'Rabies': [0.1, 0.3, 0.6],
    'Kennel Cough': [0.6, 0.3, 0.1],
    'Lyme Disease': [0.4, 0.4, 0.2],
    'Heartworm': [0.3, 0.4, 0.3],
    'Panleukopenia': [0.3, 0.4, 0.3],
    'Upper Respiratory Infection': [0.5, 0.3, 0.2],
    'Hyperthyroidism': [0.4, 0.4, 0.2]
}

# Mùa (Season) giả lập
seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
season_effects = {
    'Spring': {
        'Parvovirus': 1.1,
        'FeLV': 1.0,
        'Kennel Cough': 1.2,
        'Upper Respiratory Infection': 1.2,
        'Hyperthyroidism': 1.1
    },
    'Summer': {
        'Parvovirus': 0.9,
        'FeLV': 1.2,
        'Heartworm': 1.3,
        'Lyme Disease': 1.2,
        'Rabies': 1.4
    },
    'Autumn': {
        'Leptospirosis': 1.3,
        'Diabetes': 1.0,
        'Panleukopenia': 1.1,
        'Hyperthyroidism': 1.0
    },
    'Winter': {
        'Canine Distemper': 1.4,
        'FIV': 1.0,
        'Upper Respiratory Infection': 1.3,
        'Rabies': 1.2,
        'Panleukopenia': 1.2
    }
}

# Khu vực sống (Living Area)
living_areas = ['Urban', 'Rural']

# Triệu chứng cơ bản theo bệnh (có thể điều chỉnh theo severity)
base_symptom_probs = {
    'Parvovirus': {'Vomiting': 0.85, 'Diarrhea': 0.9, 'Appetite_Loss': 0.95, 'Fever': 0.9},
    'Canine Distemper': {'Coughing': 0.8, 'Labored_Breathing': 0.75, 'Eye_Discharge': 0.65},
    'Leptospirosis': {'Vomiting': 0.6, 'Diarrhea': 0.55, 'Lameness': 0.75},
    'FeLV': {'Appetite_Loss': 0.85, 'Weight_Loss': 0.8, 'Lethargy': 0.75},
    'FIV': {'Appetite_Loss': 0.75, 'Weight_Loss': 0.7, 'Lethargy': 0.65},
    'Diabetes': {'Appetite_Loss': 0.9, 'Weight_Loss': 0.85, 'Lethargy': 0.7},
    'Rabies': {'Aggression': 0.8, 'Lethargy': 0.7, 'Fever': 0.6},
    'Kennel Cough': {'Coughing': 0.9, 'Nasal_Discharge': 0.7},
    'Lyme Disease': {'Lameness': 0.8, 'Fever': 0.6, 'Lethargy': 0.5},
    'Heartworm': {'Coughing': 0.6, 'Labored_Breathing': 0.7, 'Lethargy': 0.8},
    'Panleukopenia': {'Vomiting': 0.7, 'Diarrhea': 0.8, 'Lethargy': 0.7},
    'Upper Respiratory Infection': {'Eye_Discharge': 0.8, 'Nasal_Discharge': 0.7, 'Coughing': 0.5},
    'Hyperthyroidism': {'Weight_Loss': 0.85, 'Appetite_Loss': 0.6, 'Lethargy': 0.4}
}

all_symptoms = ['Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing', 'Labored_Breathing',
                'Lameness', 'Skin_Lesions', 'Nasal_Discharge', 'Eye_Discharge', 'Weight_Loss', 'Fever', 'Lethargy']

def adjust_probs_by_severity(probs, severity):
    # Tăng hoặc giảm xác suất triệu chứng theo mức độ severity
    factor = {'Mild': 0.7, 'Moderate': 1.0, 'Severe': 1.3}
    new_probs = {}
    for symptom, p in probs.items():
        adjusted = p * factor[severity]
        new_probs[symptom] = min(1.0, adjusted)  # max 1.0
    return new_probs

def correlated_symptoms(symptom_dict):
    # Các tương quan triệu chứng mở rộng
    if symptom_dict.get('Vomiting') == 'Yes' and symptom_dict.get('Diarrhea') == 'No':
        if random.random() < 0.5:
            symptom_dict['Diarrhea'] = 'Yes'
    if symptom_dict.get('Appetite_Loss') == 'Yes' and symptom_dict.get('Weight_Loss') == 'No':
        if random.random() < 0.4:
            symptom_dict['Weight_Loss'] = 'Yes'
    if symptom_dict.get('Coughing') == 'Yes' and symptom_dict.get('Labored_Breathing') == 'No':
        if random.random() < 0.6:
            symptom_dict['Labored_Breathing'] = 'Yes'
    if symptom_dict.get('Eye_Discharge') == 'Yes' and symptom_dict.get('Nasal_Discharge') == 'No':
        if random.random() < 0.5:
            symptom_dict['Nasal_Discharge'] = 'Yes'
    return symptom_dict

def add_noise_numeric(value, noise_level=0.1):
    return round(value + np.random.normal(0, noise_level), 2)

def duration_category(duration):
    if duration <= 7:
        return 'Short'
    elif duration <= 20:
        return 'Medium'
    else:
        return 'Long'

# Cân bằng số lượng mẫu theo bệnh (số mẫu / bệnh)
samples_per_disease = 5000

data = []

for animal in animal_types:
    for disease in diseases[animal]:
        for _ in range(samples_per_disease):
            breed = random.choice(breeds[animal])
            gender = random.choice(genders)
            severity = random.choices(severity_levels, weights=severity_probs[disease])[0]
            season = random.choice(seasons)
            living_area = random.choice(living_areas)

            # Điều chỉnh xác suất bệnh theo mùa
            season_factor = season_effects.get(season, {}).get(disease, 1.0)

            # Tuổi, cân nặng, duration, body_temp, heart_rate theo bệnh và severity
            if animal == 'Dog':
                if disease == 'Parvovirus':
                    age = int(np.clip(np.random.normal(1.5, 1), 0.5, 3))
                    weight = round(np.clip(np.random.normal(10, 4), 5, 20), 1)
                    duration = random.randint(1, 7)
                    body_temp = np.clip(np.random.normal(39.5, 0.5), 39, 41)
                    heart_rate = random.randint(110, 140)
                elif disease == 'Canine Distemper':
                    age = int(np.clip(np.random.normal(3, 2), 1, 8))
                    weight = round(np.clip(np.random.normal(18, 6), 7, 35), 1)
                    duration = random.randint(3, 14)
                    body_temp = np.clip(np.random.normal(39.0, 0.5), 38, 40.5)
                    heart_rate = random.randint(90, 130)
                elif disease == 'Leptospirosis':
                    age = int(np.clip(np.random.normal(5, 3), 1, 12))
                    weight = round(np.clip(np.random.normal(20, 8), 10, 40), 1)
                    duration = random.randint(5, 30)
                    body_temp = np.clip(np.random.normal(38.5, 0.4), 37.5, 39.5)
                    heart_rate = random.randint(70, 120)
            else:  # Cat
                if disease in ['FeLV', 'FIV']:
                    age = int(np.clip(np.random.normal(5, 3), 2, 15))
                    weight = round(np.clip(np.random.normal(4, 2), 2, 7), 1)
                    duration = random.randint(7, 30)
                    body_temp = np.clip(np.random.normal(39.0, 0.4), 38, 40)
                    heart_rate = random.randint(100, 140)
                elif disease == 'Diabetes':
                    age = int(np.clip(np.random.normal(8, 4), 5, 15))
                    weight = round(np.clip(np.random.normal(6, 3), 3, 10), 1)
                    duration = random.randint(10, 30)
                    body_temp = np.clip(np.random.normal(38.5, 0.3), 37.5, 39)
                    heart_rate = random.randint(80, 120)

            # Áp season factor tăng giảm nhẹ xác suất triệu chứng (giả lập)
            symptom_base_probs = adjust_probs_by_severity(base_symptom_probs[disease], severity)
            symptom_base_probs = {k: min(1.0, v * season_factor) for k, v in symptom_base_probs.items()}

            # Thêm noise cho các đặc trưng số
            body_temp = add_noise_numeric(body_temp, noise_level=0.2)
            heart_rate = int(np.clip(heart_rate + np.random.normal(0, 3), 40, 200))
            weight = max(0.5, weight)

            # Khởi tạo triệu chứng all No
            symptom_data = {symptom: 'No' for symptom in all_symptoms}

            # Gán triệu chứng theo xác suất
            for symptom, prob in symptom_base_probs.items():
                symptom_data[symptom] = 'Yes' if random.random() < prob else 'No'

            # Triệu chứng liên quan bổ sung
            symptom_data = correlated_symptoms(symptom_data)

            # Thêm triệu chứng ngẫu nhiên nhỏ (noise)
            for symptom in all_symptoms:
                if symptom_data[symptom] == 'No' and random.random() < 0.02:
                    symptom_data[symptom] = 'Yes'

            # Thêm biến Duration category
            dur_cat = duration_category(duration)

            # Tạo dữ liệu dạng dictionary
            record = {
                'Animal_Type': animal,
                'Breed': breed,
                'Gender': gender,
                'Age_Years': age,
                'Weight_kg': weight,
                'Duration_Days': duration,
                'Duration_Category': dur_cat,
                'Severity': severity,
                'Season': season,
                'Living_Area': living_area,
                'Body_Temperature_C': body_temp,
                'Heart_Rate_BPM': heart_rate,
                'Disease': disease
            }
            record.update(symptom_data)

            data.append(record)

# Chuyển thành DataFrame
df = pd.DataFrame(data)

# Shuffle lại toàn bộ dữ liệu
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Xuất ra file CSV
df.to_csv('synthetic_pet_disease_data_extended_1.csv', index=False)

print('Data generation completed. Dataset shape:', df.shape)
