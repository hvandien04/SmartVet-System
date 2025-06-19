import React, { useState } from 'react';
import '../styles/modal.css';

const CreatePetModal = ({ onClose, onCreate }) => {
    const [petData, setPetData] = useState({
        name: '',
        species: '',
        breed: '',
        gender: '',
        birthDate: '',
        ownerId: '',
    });

    const handleChange = (e) => {
        const { name, value } = e.target;
        setPetData({
            ...petData,
            [name]: value,
        });
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (
            !petData.name ||
            !petData.species ||
            !petData.breed ||
            !petData.gender ||
            !petData.birthDate ||
            !petData.ownerId
        ) {
            alert('Vui lòng điền đầy đủ thông tin.');
            return;
        }
        onCreate(petData);
    };

    return (
        <div className="modal-overlay">
            <div className="modal-content">
                <h2>Tạo Thú Cưng Mới</h2>
                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label>Tên Thú Cưng:</label>
                        <input
                            type="text"
                            name="name"
                            value={petData.name}
                            onChange={handleChange}
                            placeholder="Nhập tên thú cưng"
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Loài:</label>
                        <input
                            type="text"
                            name="species"
                            value={petData.species}
                            onChange={handleChange}
                            placeholder="Nhập loài (ví dụ: Dog, Cat)"
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Giống:</label>
                        <input
                            type="text"
                            name="breed"
                            value={petData.breed}
                            onChange={handleChange}
                            placeholder="Nhập giống (ví dụ: Poodle, Persian)"
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Giới Tính:</label>
                        <select
                            name="gender"
                            value={petData.gender}
                            onChange={handleChange}
                            required
                        >
                            <option value="">Chọn giới tính</option>
                            <option value="Male">Đực</option>
                            <option value="Female">Cái</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label>Ngày Sinh:</label>
                        <input
                            type="date"
                            name="birthDate"
                            value={petData.birthDate}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Mã Chủ Sở Hữu:</label>
                        <input
                            type="text"
                            name="ownerId"
                            value={petData.ownerId}
                            onChange={handleChange}
                            placeholder="Nhập mã chủ sở hữu (ví dụ: U3A509)"
                            required
                        />
                    </div>
                    <div className="modal-actions">
                        <button type="submit" className="btn btn-primary">
                            Tạo
                        </button>
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={onClose}
                        >
                            Hủy
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default CreatePetModal;
