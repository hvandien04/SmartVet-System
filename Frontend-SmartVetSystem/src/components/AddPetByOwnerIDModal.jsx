import React, { useState } from 'react';
import '../styles/dashboard.css';

const AddPetByOwnerIDModal = ({ ownerId, onClose, onCreate }) => {
    const [petData, setPetData] = useState({
        name: '',
        birthDate: '',
        breed: '',
        species: '',
    });

    const handleChange = (e) => {
        const { name, value } = e.target;
        setPetData(prev => ({
            ...prev,
            [name]: value,
        }));
    };

    const handleSubmit = () => {
        if (!ownerId) return;
        onCreate({ ...petData, ownerId }); // gửi thêm ownerId
        onClose();
    };

    return (
        <div className="modal-overlay">
            <div className="modal">
                <h2>Thêm thú cưng mới</h2>

                <div className="form-group">
                    <label>Tên</label>
                    <input
                        name="name"
                        value={petData.name}
                        onChange={handleChange}
                        placeholder="Nhập tên thú cưng"
                    />
                </div>

                <div className="form-group">
                    <label>Ngày sinh</label>
                    <input
                        type="date"
                        name="birthDate"
                        value={petData.birthDate}
                        onChange={handleChange}
                    />
                </div>

                <div className="form-group">
                    <label>Giống</label>
                    <input
                        name="breed"
                        value={petData.breed}
                        onChange={handleChange}
                        placeholder="Nhập giống"
                    />
                </div>

                <div className="form-group">
                    <label>Loài</label>
                    <input
                        name="species"
                        value={petData.species}
                        onChange={handleChange}
                        placeholder="Nhập loài"
                    />
                </div>

                <div style={{ marginTop: '20px', textAlign: 'right' }}>
                    <button onClick={onClose}>Hủy</button>
                    <button onClick={handleSubmit}>Lưu</button>
                </div>
            </div>
        </div>
    );
};

export default AddPetByOwnerIDModal;
