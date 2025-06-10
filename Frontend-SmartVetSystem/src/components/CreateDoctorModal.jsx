import React, { useState } from 'react';
import '../styles/dashboard.css';

const CreateDoctorModal = ({ onClose, onCreate }) => {
    const [doctorData, setDoctorData] = useState({
        username: '',
        password: '12345678@aA',
        fullName: '',
        email: '',
        phone: '',
        address: '',
        role: 'DOCTOR',
    });



    const handleChange = (e) => {
        const { name, value, checked } = e.target;

        if (name === 'isAdmin') {
            setDoctorData(prev => ({
                ...prev,
                role: checked ? 'ADMIN' : 'DOCTOR', // xử lý role dựa vào checkbox
            }));
        } else {
            setDoctorData(prev => ({
                ...prev,
                [name]: value,
            }));
        }
    };


    const handleSubmit = () => {
        onCreate(doctorData);
        onClose();
    };

    return (
        <div className="modal-overlay">
            <div className="modal">
                <h2>🩺 Hồ sơ bác sĩ</h2>

                <div className="form-group">
                    <label>Username</label>
                    <input name="username" value={doctorData.username} onChange={handleChange} placeholder="Doctor username"/>
                </div>

                <div className="form-group">
                    <label>Full name</label>
                    <input name="fullName" value={doctorData.fullName} onChange={handleChange} placeholder="Doctor full name" />
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
                        name="isAdmin"
                        checked={doctorData.role === 'ADMIN'}
                        onChange={handleChange}
                    />
                    Tài khoản này được cấp quyền admin?
                </label>


                <div style={{ marginTop: '20px', textAlign: 'right' }}>
                    <button onClick={onClose}>Cancel</button>
                    <button onClick={handleSubmit}>Save profile</button>
                </div>
            </div>
        </div>
    );
};

export default CreateDoctorModal;
