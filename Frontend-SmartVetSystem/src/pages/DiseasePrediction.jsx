import React, { useEffect, useState } from "react";
import Sidebar from "../components/Sidebar";
import { getSidebarGroups } from "../components/sidebarData";
import { useAuth } from "../context/AuthContext";
import SearchBox from "../components/SearchBox";
import ChatbotPanel from "../components/ChatbotPanel";
import "../styles/diseaseprediction.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

const DiseasePrediction = () => {
    /* ---------- Sidebar ---------- */
    const { user } = useAuth();
    const [sidebarGroups, setSidebarGroups] = useState([]);
    const [activeSidebarItem, setActiveSidebarItem] = useState("Dự đoán bệnh");
    useEffect(() => {
        setSidebarGroups(getSidebarGroups(user || null));
    }, [user]);

    /* ---------- Form options ---------- */
    const [formOpts, setFormOpts] = useState(null); // {animalTypes, breeds,...}
    const [breeds, setBreeds] = useState([]);

    useEffect(() => {
        fetch(`${API_URL}/api/form-options`)
            .then((r) => r.json())
            .then((data) => {
                setFormOpts(data);

                const firstAnimal = data.animalTypes?.[0] ?? "";
                const firstBreed =
                    data.breeds?.[firstAnimal]?.[0] ?? "";

                setBreeds(data.breeds[firstAnimal] || []);

                setForm((f) => ({
                    ...f,
                    animalType: firstAnimal,
                    breed: firstBreed,
                    durationCategory: data.durationCategories?.[0] ?? "",
                    severity: data.severities?.[0] ?? "",
                    season: data.seasons?.[0] ?? "",
                    livingArea: data.livingAreas?.[0] ?? "",
                }));
            })
            .catch((err) => console.error("Lỗi lấy form-options:", err));
    }, []);

    /* ---------- Form state ---------- */
    const [form, setForm] = useState({
        animalType: "",
        breed: "",
        gender: "Male",
        age: 1,
        weight: "",
        durationDays: 1,
        severity: "",
        season: "",
        livingArea: "",
        temperature: "",
        heartRate: "",
        description: "",
    });

    /* ---------- Update breeds khi animalType đổi ---------- */
    useEffect(() => {
        if (!formOpts) return;
        const b = formOpts.breeds[form.animalType] || [];
        setBreeds(b);
        setForm((f) => ({ ...f, breed: b[0] || "" }));
    }, [form.animalType, formOpts]);

    /* ---------- Symptom Tag ---------- */
    const symptomOptions = formOpts?.symptoms || [];
    const [selectedSymptoms, setSelectedSymptoms] = useState([]);

    /* ---------- Prediction & Chat ---------- */
    const [prediction, setPrediction] = useState(null);


    /* ---------- Helpers ---------- */
    const handleChange = (field) => (e) =>
        setForm((prev) => ({ ...prev, [field]: e.target.value }));

    const toggleSymptom = (sym) => {
        setSelectedSymptoms((prev) => {
            const isSel = prev.includes(sym);
            const next = isSel ? prev.filter((s) => s !== sym) : [...prev, sym];

            setForm((f) => {
                let desc = f.description || "";
                if (!isSel) {
                    if (!desc.includes(sym)) desc += (desc ? ", " : "") + sym;
                } else {
                    const regex = new RegExp(`(,?\\s*${sym})`);
                    desc = desc.replace(regex, "").replace(/^,\\s*|\\s*,\\s*$/, "").trim();
                }
                return { ...f, description: desc };
            });
            return next;
        });
    };

    // Hàm đóng gói dữ liệu gửi API
    const buildPayload = () => {
        const payload = {
            Animal_Type: form.animalType,
            Breed: form.breed,
            Gender: form.gender,
            Age_Years: Number(form.age),
            Weight_kg: Number(form.weight),
            Duration_Days: Number(form.durationDays),
            Severity: form.severity,
            Season: form.season,
            Living_Area: form.livingArea,
            Body_Temperature_C: Number(form.temperature),
            Heart_Rate_BPM: Number(form.heartRate)
        };

        symptomOptions.forEach((col) => {
            payload[col] = selectedSymptoms.includes(col) ? 1 : 0;
        });


        return payload;
    };


    const handlePredict = async (e) => {
        e.preventDefault();
        try {
            const response = await fetch(`${API_URL}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(buildPayload())
            });

            if (!response.ok) throw new Error(await response.text());
            const data = await response.json();   // {predicted_disease, probabilities}

            setPrediction({
                disease: data.predicted_disease,
                accuracy: Math.round(
                    (data.probabilities[data.predicted_disease] || 0) * 100
                )
            });
        } catch (err) {
            console.error("Lỗi dự đoán:", err);
            alert("Không thể dự đoán – kiểm tra lại server.");
        }
    };

    /* ---------- JSX ---------- */
    return (
        <div style={{ display: "flex", height: "100dvh" }}>
            {/* Sidebar */}
            <Sidebar
                groups={sidebarGroups}
                activeItem={activeSidebarItem}
                onItemClick={(i) => setActiveSidebarItem(i)}
            />

            {/* Main content */}
            <div className="main prediction-page">
                <SearchBox />
                <h1 className="page-title">Chẩn đoán bệnh</h1>
                <p className="sub-text">
                    Vui lòng nhập các thông tin chính liên quan để dự đoán triệu chứng
                    bệnh cho thú cưng
                </p>

                <div className="prediction-wrapper">
                    {/* ---------- FORM ---------- */}
                    <form className="diagnosis-form" onSubmit={handlePredict}>
                        {/* Thông tin cơ bản */}
                        <section>
                            <h3>Thông tin cơ bản</h3>
                            <div className="grid-2-cols">
                                <label>
                                    Animal type
                                    <select
                                        value={form.animalType}
                                        onChange={handleChange("animalType")}
                                    >
                                        {(formOpts?.animalTypes || []).map((t) => (
                                            <option key={t}>{t}</option>
                                        ))}
                                    </select>
                                </label>
                                <label>
                                    Breed
                                    <select value={form.breed} onChange={handleChange("breed")}>
                                        {breeds.map((b) => (
                                            <option key={b}>{b}</option>
                                        ))}
                                    </select>
                                </label>
                                <label>
                                    Gender
                                    <select value={form.gender} onChange={handleChange("gender")}>
                                        <option>Male</option>
                                        <option>Female</option>
                                    </select>
                                </label>
                                <label>
                                    Age
                                    <input
                                        type="number"
                                        min="0"
                                        value={form.age}
                                        onChange={handleChange("age")}
                                    />
                                </label>
                                <label>
                                    Weight (kg)
                                    <input
                                        type="number"
                                        step="0.01"
                                        min="0"
                                        value={form.weight}
                                        onChange={handleChange("weight")}
                                    />
                                </label>
                            </div>
                        </section>

                        {/* Thông tin bệnh lý */}
                        <section>
                            <h3>Thông tin bệnh lý</h3>
                            <div className="grid-3-cols">
                                <label>
                                    Duration Days
                                    <input
                                        type="number"
                                        min="0"
                                        value={form.durationDays}
                                        onChange={handleChange("durationDays")}
                                    />
                                </label>
                                <label>
                                    Severity
                                    <select
                                        value={form.severity}
                                        onChange={handleChange("severity")}
                                    >
                                        {(formOpts?.severities || []).map((s) => (
                                            <option key={s}>{s}</option>
                                        ))}
                                    </select>
                                </label>
                                <label>
                                    Season
                                    <select value={form.season} onChange={handleChange("season")}>
                                        {(formOpts?.seasons || []).map((s) => (
                                            <option key={s}>{s}</option>
                                        ))}
                                    </select>
                                </label>
                                <label>
                                    Living Area
                                    <select
                                        value={form.livingArea}
                                        onChange={handleChange("livingArea")}
                                    >
                                        {(formOpts?.livingAreas || []).map((l) => (
                                            <option key={l}>{l}</option>
                                        ))}
                                    </select>
                                </label>
                                <label>
                                    Body Temperature (°C)
                                    <input
                                        type="number"
                                        step="0.01"
                                        value={form.temperature}
                                        onChange={handleChange("temperature")}
                                    />
                                </label>
                                <label>
                                    Heart Rate (BPM)
                                    <input
                                        type="number"
                                        value={form.heartRate}
                                        onChange={handleChange("heartRate")}
                                    />
                                </label>
                            </div>
                        </section>

                        {/* Mô tả + Tag */}
                        <section>
                            <h3>Mô tả & triệu chứng</h3>
                            <div className="description-tags-wrapper">
                                <textarea
                                    rows="4"
                                    placeholder="Mô tả triệu chứng..."
                                    value={form.description}
                                    onChange={handleChange("description")}
                                />
                                <div className="tag-container">
                                    {symptomOptions.map((col) => (
                                        <button
                                            key={col}
                                            type="button"
                                            onClick={() => toggleSymptom(col)}
                                            className={`symptom-tag ${selectedSymptoms.includes(col) ? "selected" : ""}`}
                                        >
                                            + {col.replaceAll("_", " ")}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </section>

                        <button className="predict-btn" type="submit">
                            Dự đoán
                        </button>

                        {/* Prediction result */}
                        {prediction && (
                            <div className="result-box">
                                <div className="result-icon">🐾</div>
                                <div className="result-text">
                                    <p>
                                        Thú cưng của bạn có thể mắc:{" "}
                                        <strong>{prediction.disease}</strong>
                                    </p>
                                    <p>Độ chính xác: {prediction.accuracy}%</p>
                                </div>
                            </div>
                        )}
                    </form>

                    {/* ---------- CHAT ---------- */}
                    <ChatbotPanel />
                </div>
            </div>
        </div>
    );
};

export default DiseasePrediction;
