import React from 'react';

const HistoryDetail = ({ isOpen, onClose, data }) => {
    if (!isOpen || !data) return null;

    const overlayStyle = {
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.4)',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 999,
    };

    const modalStyle = {
        backgroundColor: '#fff',
        padding: '20px',
        borderRadius: '10px',
        width: '600px',
        maxHeight: '80vh',
        overflow: 'hidden',
        boxShadow: '0 5px 15px rgba(0,0,0,0.3)',
    };

    const scrollAreaStyle = {
        maxHeight: '65vh',
        overflowY: 'auto',
        marginTop: '10px',
        paddingRight: '10px',
        textAlign: 'left',
    };

    const formGroupStyle = {
        marginBottom: '12px',
        padding: '10px',
        backgroundColor: '#f9f9f9',
        borderRadius: '6px',
    };

    const labelStyle = {
        fontWeight: 'bold',
        marginRight: '6px',
        color: '#333',
    };

    return (
        <div style={overlayStyle} onClick={onClose}>
            <div style={modalStyle} onClick={(e) => e.stopPropagation()}>
                <h2>🔍 Thông tin dự đoán bệnh</h2>

                <div style={scrollAreaStyle}>
                    {Object.entries({
                        'Mã Dự Đoán': data.predictId,
                        'Bác Sĩ': data.doctorName,
                        'Thời Gian': data.createdAt,
                        'Giống Loài': data.species,
                        'Giới Tính': data.gender,
                        'Tuổi': data.age,
                        'Cân Nặng': `${data.weight} kg`,
                        'Số Ngày Phát Bệnh': data.daysSick,
                        'Thời Gian Bệnh': data.timeCategory,
                        'Mức Độ Nghiêm Trọng': data.severity,
                        'Mùa': data.season,
                        'Khu Vực Sống': data.location,
                        'Nhiệt Độ Cơ Thể': `${data.temperature} °C`,
                        'Nhịp Tim': `${data.heartRate} bpm`,
                        'Mô Tả Triệu Chứng': data.symptoms,
                        'Bệnh Dự Đoán': data.predictedDisease,
                        'Xác Suất': `${data.probability}%`,
                    }).map(([label, value]) => (
                        <div style={formGroupStyle} key={label}>
                            <label style={labelStyle}>{label}:</label>
                            <span>{value || 'N/A'}</span>
                        </div>
                    ))}
                </div>

                <div style={{ textAlign: 'right', marginTop: '20px' }}>
                    <button onClick={onClose}>Đóng</button>
                </div>
            </div>
        </div>
    );
};

export default HistoryDetail;
