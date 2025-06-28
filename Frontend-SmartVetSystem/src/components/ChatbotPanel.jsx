import React, { useState, useRef, useEffect } from "react";
import "../styles/diseaseprediction.css";

// Map cột nhị phân mà backend yêu cầu
const binaryCols = [
    "Appetite_Loss",
    "Vomiting",
    "Diarrhea",
    "Coughing",
    "Labored_Breathing",
    "Lameness",
    "Skin_Lesions",
    "Nasal_Discharge",
    "Eye_Discharge",
    "Weight_Loss",
    "Fever",
    "Lethargy",
];

/* ---------- Component con để hiển thị form triệu chứng ---------- */
function SymptomForm({ instruction, symptoms, onSubmit, disabled }) {
    const [selected, setSelected] = useState([]);

    const toggle = (val) => {
        if (disabled) return;
        setSelected((prev) =>
            prev.includes(val) ? prev.filter((i) => i !== val) : [...prev, val]
        );
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (disabled) return;
        onSubmit(selected);
    };

    return (
        <form className="symptom-form" onSubmit={handleSubmit}>
            <p>{instruction || "Vui lòng chọn triệu chứng phù hợp:"}</p>
            {symptoms.map((q, idx) => (
                <label key={idx} style={{ display: "block", opacity: disabled ? 0.9 : 1 }}>
                    <input
                        type="checkbox"
                        onChange={() => toggle(binaryCols[idx])}
                        checked={selected.includes(binaryCols[idx])}
                        disabled={disabled}
                    />{" "}
                    {q}
                </label>
            ))}
            {!disabled && (
                <button type="submit">Gửi</button>
            )}
        </form>
    );
}

/* ---------- Component chính ---------- */
export default function ChatbotPanel() {
    /* ----- State hiển thị ----- */
    const [messages, setMessages] = useState([]); // [{type:'text'|'symptom', from:'user'|'bot', text:'', symptoms:[] }]
    const [chatInput, setChatInput] = useState("");
    const [hasInteracted, setHasInteracted] = useState(false);

    /* ----- State logic hội thoại ----- */
    const [followUpQuestions, setFollowUpQuestions] = useState([]);
    const [answers, setAnswers] = useState({});
    const [waitingForFollowUp, setWaitingForFollowUp] = useState(false);
    const [initialInput, setInitialInput] = useState(null);
    const [hasAskedFollowUp, setHasAskedFollowUp] = useState(false);

    const chatEndRef = useRef(null);

    /* ----- Tự cuộn cuối chat ----- */
    useEffect(() => {
        if (hasInteracted) {
            chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
        }
    }, [messages]);

    /* ---------- Helpers ---------- */
    const addMessage = (msg) => {
        setMessages((prev) => [
            ...prev,
            { ...msg, id: Date.now().toString() + Math.random().toString(36).slice(2) },
        ]);
        setHasInteracted(true);
    };

    const addLoading = () => addMessage({ type: "text", from: "bot", text: "⏳ Đang xử lý...", loading: true });
    const removeLastLoading = () =>
        setMessages((prev) =>
            prev.filter((m, idx) => !(idx === prev.length - 1 && m.loading))
        );

    const askNextQuestion = () => {
        if (followUpQuestions.length > 0) {
            addMessage({
                type: "text",
                from: "bot",
                text: "📋 " + followUpQuestions[0],
            });
        }
    };

    useEffect(() => {
        if (followUpQuestions.length > 0 && waitingForFollowUp && !hasAskedFollowUp) {
            askNextQuestion();
            setHasAskedFollowUp(true);
        }
    }, [followUpQuestions, waitingForFollowUp, hasAskedFollowUp]);

    /* ---------- Gửi dữ liệu follow-up ---------- */
    const sendFollowUpAnswers = async () => {
        addLoading();
        try {
            const res = await fetch("http://localhost:8000/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ description: initialInput, answers }),
            });
            await handlePredictionResponse(res);
        } catch (err) {
            removeLastLoading();
            addMessage({
                type: "text",
                from: "bot",
                text: "🚫 Lỗi khi gọi API: " + err.message,
            });
        }
    };

    /* ---------- Xử lý phản hồi backend ---------- */
    const handlePredictionResponse = async (response) => {
        removeLastLoading();
        const result = await response.json();

        if (result.error) {
            addMessage({ type: "text", from: "bot", text: "❌ Lỗi: " + result.error });
            return;
        }

        /* --- 1. Có kết quả dự đoán --- */
        if (result.prediction) {
            addMessage({
                type: "text",
                from: "bot",
                text:
                    "🎯 Kết quả dự đoán: " +
                    result.prediction +
                    (result.confidence !== undefined
                        ? ` (Độ tin cậy: ${(result.confidence * 100).toFixed(2)}%)`
                        : ""),
            });
            /* reset luồng */
            setFollowUpQuestions([]);
            setAnswers({});
            setInitialInput(null);
            setWaitingForFollowUp(false);
            setHasAskedFollowUp(false)
            return;
        }

        /* --- 2. Backend hỏi thêm câu hỏi tự do --- */
        if (result.questions) {
            setFollowUpQuestions(result.questions.slice());
            setWaitingForFollowUp(true);
            return;
        }

        /* --- 3. Backend yêu cầu xác nhận triệu chứng --- */
        if (result.ask_symptom_confirmation && result.symptoms) {
            addMessage({
                type: "symptom",
                from: "bot",
                symptoms: result.symptoms,
                instruction: result.message,
            });
            setWaitingForFollowUp(true);
            return;
        }

        /* --- 4. Không có gì --- */
        addMessage({ type: "text", from: "bot", text: "⚠️ Không có kết quả." });
    };

    /* ---------- Sự kiện nhấn gửi ---------- */
    const onSubmit = async (e) => {
        e.preventDefault();
        const input = chatInput.trim();
        if (!input) return;

        /* --- Trả lời follow-up --- */
        if (waitingForFollowUp && followUpQuestions.length > 0) {
            const current = followUpQuestions[0];
            setAnswers((prev) => ({ ...prev, [current]: input }));
            addMessage({ type: "text", from: "user", text: input });

            setChatInput("");
            const remain = followUpQuestions.slice(1);
            setFollowUpQuestions(remain);

            if (remain.length === 0) {
                setWaitingForFollowUp(false);
                setHasAskedFollowUp(false);
                await sendFollowUpAnswers();
            } else {
                setHasAskedFollowUp(false);
            }

            return;
        }

        /* --- Tin nhắn đầu tiên */
        addMessage({ type: "text", from: "user", text: input });
        if (!initialInput) setInitialInput(input);
        setChatInput("");

        addLoading();
        try {
            const res = await fetch("http://localhost:8000/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ description: initialInput || input, answers }),
            });
            await handlePredictionResponse(res);
        } catch (err) {
            removeLastLoading();
            addMessage({
                type: "text",
                from: "bot",
                text: "🚫 Lỗi khi gọi API: " + err.message,
            });
        }
    };

    /* ---------- Xử lý form triệu chứng ---------- */
    const handleSymptomSubmit = async (selectedSymptoms) => {
        let updatedAnswers = {};

        if (selectedSymptoms.length === 0) {
            updatedAnswers = { ...answers, SYMPTOMS_UNKNOWN: 1 };
        } else {
            updatedAnswers = selectedSymptoms.reduce(
                (acc, s) => ({ ...acc, [s]: 1 }),
                { ...answers }
            );
        }

        setAnswers(updatedAnswers);

        // Cập nhật message type=symptom để nó thành disabled
        setMessages((prev) =>
            prev.map((m) =>
                m.type === "symptom" && !m.disabled ? { ...m, disabled: true } : m
            )
        );

        // Gửi dữ liệu
        addLoading();
        try {
            const res = await fetch("http://localhost:8000/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    description: initialInput,
                    answers: updatedAnswers,
                }),
            });
            await handlePredictionResponse(res);
        } catch (err) {
            removeLastLoading();
            addMessage({
                type: "text",
                from: "bot",
                text: "🚫 Lỗi khi gọi API: " + err.message,
            });
        }
    };



    /* ---------- JSX ---------- */
    return (
        <div className="chatbot-panel">
            <header>AI CHAT BOT</header>

            <div className="chat-history">
                {messages.map((m) =>
                    m.type === "symptom" ? (
                        <SymptomForm
                            key={m.id}
                            instruction={m.instruction}
                            symptoms={m.symptoms}
                            onSubmit={handleSymptomSubmit}
                            disabled={m.disabled}
                        />
                    ) : (
                        <div
                            key={m.id}
                            className={`chat-message ${m.from === "user" ? "user" : "bot"}`}
                        >
                            {m.text}
                        </div>
                    )
                )}
                <div ref={chatEndRef} />
            </div>

            <form className="chat-input" onSubmit={onSubmit}>
        <textarea
            value={chatInput}
            onChange={(e) => {
                /* tự co giãn chiều cao */
                e.target.style.height = "auto";
                e.target.style.height = e.target.scrollHeight + "px";
                setChatInput(e.target.value);
            }}
            placeholder="Nhập tin nhắn..."
            rows={1}
        />
                <button className="send-btn" type="submit">
                    ➤
                </button>
            </form>
        </div>
    );
}
