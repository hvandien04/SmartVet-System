// MedicalRecordDetail.jsx
import React, { useState, useEffect } from 'react';
import '../styles/dashboard.css';

const MedicalRecordDetail = ({ isOpen, onClose, onSave, data }) => {
    const [formData, setFormData] = useState({
        petId: '',
        userId: '',
        visitDate: '',
        symptoms: '',
        diagnosisSummary: '',
        clinicalTestResults: '',
        animalType: '',
        breed: '',
        gender: '',
        ageYears: '',
        weightKg: '',
        durationDays: '',
        durationCategory: '',
        severity: '',
        season: '',
        livingArea: '',
        bodyTemperatureC: '',
        heartRateBpm: '',
        description: '',
        treatmentPlan: '',
        medicationsPrescribed: '',
        followUpRequired: false,
        nextVisitDate: '',
        noteForOwner: '',
        status: '',
        medicalImageRequest: [],
    });

    useEffect(() => {
        if (isOpen && data) {
            console.log('MedicalRecordDetail received data:', data);
            const newFormData = {
                petId: data.petId || '',
                userId: data.userId || '',
                visitDate: data.visitDate ? new Date(data.visitDate).toISOString().split('T')[0] : '',
                symptoms: data.symptoms || '',
                diagnosisSummary: data.diagnosisSummary || '',
                clinicalTestResults: data.clinicalTestResults || '',
                animalType: data.animalType || '',
                breed: data.breed || '',
                gender: data.gender || '', // Đảm bảo gender được điền
                ageYears: data.ageYears || '',
                weightKg: data.weightKg || '',
                durationDays: data.durationDays || '',
                durationCategory: data.durationCategory || '',
                severity: data.severity || '',
                season: data.season || '',
                livingArea: data.livingArea || '',
                bodyTemperatureC: data.bodyTemperatureC || '',
                heartRateBpm: data.heartRateBpm || '',
                description: data.description || '',
                treatmentPlan: data.treatmentPlan || '',
                medicationsPrescribed: data.medicationsPrescribed || '',
                followUpRequired: data.followUpRequired || false,
                nextVisitDate: data.nextVisitDate ? new Date(data.nextVisitDate).toISOString().split('T')[0] : '',
                noteForOwner: data.noteForOwner || '',
                status: data.status || '',
                medicalImageRequest: data.medicalImageRequest || [],
            };
            setFormData(newFormData);
            console.log('Form data set to:', newFormData);
        } else if (!isOpen) {
            setFormData({
                petId: '',
                userId: '',
                visitDate: '',
                symptoms: '',
                diagnosisSummary: '',
                clinicalTestResults: '',
                animalType: '',
                breed: '',
                gender: '',
                ageYears: '',
                weightKg: '',
                durationDays: '',
                durationCategory: '',
                severity: '',
                season: '',
                livingArea: '',
                bodyTemperatureC: '',
                heartRateBpm: '',
                description: '',
                treatmentPlan: '',
                medicationsPrescribed: '',
                followUpRequired: false,
                nextVisitDate: '',
                noteForOwner: '',
                status: '',
                medicalImageRequest: [],
            });
        }
    }, [data, isOpen]);

    const handleChange = (e) => {
        const { name, value, type, checked } = e.target;
        setFormData((prev) => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : value,
        }));
    };

    const handleSave = () => {
        if (!formData.petId) {
            alert('Vui lòng nhập Pet ID');
            return;
        }
        if (!formData.userId) {
            alert('Vui lòng nhập User ID');
            return;
        }
        if (!formData.visitDate) {
            alert('Vui lòng nhập ngày khám');
            return;
        }
        console.log('Saving form data:', formData);
        onSave({
            ...formData,
            durationDays: parseInt(formData.durationDays) || 0,
            ageYears: parseFloat(formData.ageYears) || 0,
            weightKg: parseFloat(formData.weightKg) || 0,
            bodyTemperatureC: parseFloat(formData.bodyTemperatureC) || 0,
            heartRateBpm: parseInt(formData.heartRateBpm) || 0,
            visitDate: formData.visitDate ? `${formData.visitDate}T00:00:00.000Z` : null,
            nextVisitDate: formData.nextVisitDate ? `${formData.nextVisitDate}T00:00:00.000Z` : null,
            medicalImageRequest: formData.medicalImageRequest || [],
        });
        onClose();
    };

    if (!isOpen) return null;

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                <h2>📋 Chi tiết Hồ sơ Y tế</h2>
                <div className="modal-scroll" style={{ maxHeight: '70vh', overflowY: 'auto' }}>
                    <div className="form-group">
                        <label>Pet ID:</label>
                        <input type="text" name="petId" value={formData.petId} onChange={handleChange} placeholder="Nhập Pet ID" />
                    </div>
                    <div className="form-group">
                        <label>User ID:</label>
                        <input type="text" name="userId" value={formData.userId} onChange={handleChange} placeholder="Nhập User ID" />
                    </div>
                    <div className="form-group">
                        <label>Ngày khám:</label>
                        <input type="date" name="visitDate" value={formData.visitDate} onChange={handleChange} />
                    </div>
                    <div className="form-group">
                        <label>Triệu chứng:</label>
                        <textarea name="symptoms" value={formData.symptoms} onChange={handleChange} placeholder="Nhập triệu chứng" />
                    </div>
                    <div className="form-group">
                        <label>Tóm tắt chẩn đoán:</label>
                        <textarea name="diagnosisSummary" value={formData.diagnosisSummary} onChange={handleChange} placeholder="Nhập tóm tắt chẩn đoán" />
                    </div>
                    <div className="form-group">
                        <label>Kết quả xét nghiệm:</label>
                        <textarea name="clinicalTestResults" value={formData.clinicalTestResults} onChange={handleChange} placeholder="Nhập kết quả xét nghiệm" />
                    </div>
                    <div className="form-group">
                        <label>Loại động vật:</label>
                        <input type="text" name="animalType" value={formData.animalType} onChange={handleChange} placeholder="Nhập loại động vật" maxLength={50} />
                    </div>
                    <div className="form-group">
                        <label>Giống:</label>
                        <input type="text" name="breed" value={formData.breed} onChange={handleChange} placeholder="Nhập giống" maxLength={50} />
                    </div>
                    <div className="form-group">
                        <label>Giới tính:</label>
                        <select name="gender" value={formData.gender} onChange={handleChange}>
                            <option value="">Chọn giới tính</option>
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                            <option value="Unknown">Unknown</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label>Tuổi (năm):</label>
                        <input type="number" name="ageYears" value={formData.ageYears} onChange={handleChange} step="0.1" placeholder="Nhập tuổi" />
                    </div>
                    <div className="form-group">
                        <label>Cân nặng (kg):</label>
                        <input type="number" name="weightKg" value={formData.weightKg} onChange={handleChange} step="0.1" placeholder="Nhập cân nặng" />
                    </div>
                    <div className="form-group">
                        <label>Thời gian bệnh (ngày):</label>
                        <input type="number" name="durationDays" value={formData.durationDays} onChange={handleChange} placeholder="Nhập số ngày" />
                    </div>
                    <div className="form-group">
                        <label>Thể loại thời gian:</label>
                        <select name="durationCategory" value={formData.durationCategory} onChange={handleChange}>
                            <option value="">Chọn thể loại</option>
                            <option value="Short">Short</option>
                            <option value="Acute">Acute</option>
                            <option value="Chronic">Chronic</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label>Mức độ nghiêm trọng:</label>
                        <select name="severity" value={formData.severity} onChange={handleChange}>
                            <option value="">Chọn mức độ</option>
                            <option value="Mild">Mild</option>
                            <option value="Moderate">Moderate</option>
                            <option value="Severe">Severe</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label>Mùa:</label>
                        <select name="season" value={formData.season} onChange={handleChange}>
                            <option value="">Chọn mùa</option>
                            <option value="Spring">Spring</option>
                            <option value="Summer">Summer</option>
                            <option value="Fall">Fall</option>
                            <option value="Winter">Winter</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label>Khu vực sống:</label>
                        <select name="livingArea" value={formData.livingArea} onChange={handleChange}>
                            <option value="">Chọn khu vực</option>
                            <option value="Urban">Urban</option>
                            <option value="Rural">Rural</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label>Nhiệt độ cơ thể (°C):</label>
                        <input type="number" name="bodyTemperatureC" value={formData.bodyTemperatureC} onChange={handleChange} step="0.1" placeholder="Nhập nhiệt độ" />
                    </div>
                    <div className="form-group">
                        <label>Nhịp tim (bpm):</label>
                        <input type="number" name="heartRateBpm" value={formData.heartRateBpm} onChange={handleChange} placeholder="Nhập nhịp tim" />
                    </div>
                    <div className="form-group">
                        <label>Mô tả:</label>
                        <textarea name="description" value={formData.description} onChange={handleChange} placeholder="Nhập mô tả" maxLength={255} />
                    </div>
                    <div className="form-group">
                        <label>Kế hoạch điều trị:</label>
                        <textarea name="treatmentPlan" value={formData.treatmentPlan} onChange={handleChange} placeholder="Nhập kế hoạch điều trị" maxLength={255} />
                    </div>
                    <div className="form-group">
                        <label>Thuốc được kê:</label>
                        <textarea name="medicationsPrescribed" value={formData.medicationsPrescribed} onChange={handleChange} placeholder="Nhập thuốc được kê" maxLength={255} />
                    </div>
                    <div className="form-group">
                        <label>Cần theo dõi:</label>
                        <input type="checkbox" name="followUpRequired" checked={formData.followUpRequired} onChange={handleChange} />
                    </div>
                    <div className="form-group">
                        <label>Ngày tái khám:</label>
                        <input type="date" name="nextVisitDate" value={formData.nextVisitDate} onChange={handleChange} />
                    </div>
                    <div className="form-group">
                        <label>Ghi chú cho chủ nuôi:</label>
                        <textarea name="noteForOwner" value={formData.noteForOwner} onChange={handleChange} placeholder="Nhập ghi chú" maxLength={255} />
                    </div>
                    <div className="form-group">
                        <label>Trạng thái:</label>
                        <select name="status" value={formData.status} onChange={handleChange}>
                            <option value="">Chọn trạng thái</option>
                            <option value="ongoing">Ongoing</option>
                            <option value="completed">Completed</option>
                            <option value="cancelled">Cancelled</option>
                        </select>
                    </div>
                </div>
                <div style={{ textAlign: 'right', marginTop: '20px' }}>
                    <button onClick={onClose} style={{ marginRight: '10px' }}>Hủy</button>
                    <button onClick={handleSave}>Lưu hồ sơ</button>
                </div>
            </div>
        </div>
    );
};

export default MedicalRecordDetail;