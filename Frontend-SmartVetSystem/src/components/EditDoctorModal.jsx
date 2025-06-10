import React, { useState, useEffect } from 'react';
import '../styles/dashboard.css';

const EditDoctorModal = ({ doctor, onClose, onUpdate }) => {
    const [doctorData, setDoctorData] = useState({
        email: '',
        password: '',
        fullName: '',
        phone: '',
        address: '',
        role: 'DOCTOR',
    });

    useEffect(() => {
        if (doctor) {
            setDoctorData({
                email: doctor.email || '',
                password: '',
                fullName: doctor.fullName || '',
                phone: doctor.phone || '',
                address: doctor.address || '',
                role: doctor.role ? doctor.role : 'DOCTOR',
            });
        }
    }, [doctor]);

    const handleChange = (e) => {
        const { name, value, checked } = e.target;

        if (name === 'role') {
            setDoctorData(prev => ({
                ...prev,
                role: checked ? 'ADMIN' : 'DOCTOR',
            }));
        } else {
            setDoctorData(prev => ({
                ...prev,
                [name]: value,
            }));
        }
    };

    const handleSubmit = () => {
        let passwordToSend = doctorData.password.trim();
        if (!passwordToSend) {
            passwordToSend = '12345678@aA';
        }

        // Gửi đúng giá trị role hiện tại trong state
        const dataToUpdate = {
            ...doctorData,
            password: passwordToSend,
            role: doctorData.role === 'DOCTOR' ? 'DOCTOR' : 'ADMIN',
        };

        onUpdate(doctor.userId, dataToUpdate);
        onClose();
    };



    return (
        <div className="modal-overlay">
            <div className="modal">
                <h2>✏️ Chỉnh sửa thông tin bác sĩ</h2>

                <div className="form-group">
                    <label>Full name</label>
                    <input name="fullName" value={doctorData.fullName} onChange={handleChange} placeholder="Doctor full name" />
                </div>

                <div className="form-group">
                    <label>New Password (bỏ trống nếu chuyển về mặc định)</label>
                    <input
                        name="password"
                        type="password"
                        value={doctorData.password}
                        onChange={handleChange}
                        placeholder="Enter new password"
                    />
                </div>

                <div className="form-group">
                    <label>Email</label>
                    <input name="email" value={doctorData.email} onChange={handleChange} placeholder="Doctor email" />
                </div>

                <div className="form-group">
                    <label>Phone number</label>
                    <input name="phone" value={doctorData.phone} onChange={handleChange} placeholder="Doctor phone number" />
                </div>

                <div className="form-group">
                    <label>Address</label>
                    <input name="address" value={doctorData.address} onChange={handleChange} placeholder="Doctor address" />
                </div>

                <label className="checkbox-label">
                    <input
                        type="checkbox"
                        name="role"
                        checked={doctorData.role === 'ADMIN'}
                        onChange={handleChange}
                    />
                    Tài khoản này được cấp quyền admin?
                </label>

                <div style={{ marginTop: '20px', textAlign: 'right' }}>
                    <button onClick={onClose}>Cancel</button>
                    <button onClick={handleSubmit}>Update profile</button>
                </div>
            </div>
        </div>
    );
};

export default EditDoctorModal;
