// EditPetModal.jsx
import React, { useState, useEffect } from 'react';
import '../styles/modal.css';
const EditPetModal = ({ pet, onClose, onUpdate }) => {
    const [petData, setPetData] = useState({
        petId: '',
        name: '',
        species: '',
        breed: '',
        gender: '',
        birthDate: '',
        owner: { ownerId: '' },
    });

    // Điền sẵn dữ liệu thú cưng khi modal mở
    useEffect(() => {
        if (pet) {
            setPetData({
                petId: pet.petId || '',
                name: pet.name || '',
                species: pet.species || '',
                breed: pet.breed || '',
                gender: pet.gender || '',
                birthDate: pet.birthDate || '',
                owner: { ownerId: pet.owner?.ownerId || '' },
            });
        }
    }, [pet]);

    const handleChange = (e) => {
        const { name, value } = e.target;
        if (name === 'ownerId') {
            setPetData({
                ...petData,
                owner: { ...petData.owner, ownerId: value },
            });
        } else {
            setPetData({
                ...petData,
                [name]: value,
            });
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        // Đảm bảo tất cả các trường bắt buộc đã được điền
        if (
            !petData.petId ||
            !petData.name ||
            !petData.species ||
            !petData.breed ||
            !petData.gender ||
            !petData.birthDate ||
            !petData.owner.ownerId
        ) {
            alert('Vui lòng điền đầy đủ thông tin.');
            return;
        }
        onUpdate(petData);
    };

    return (
        <div className="modal-overlay">
            <div className="modal-content">
                <h2>Chỉnh Sửa Thú Cưng</h2>
                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label>Mã Thú Cưng:</label>
                        <input
                            type="text"
                            name="petId"
                            value={petData.petId}
                            onChange={handleChange}
                            placeholder="Nhập mã thú cưng"
                            disabled // Không cho chỉnh sửa petId
                        />
                    </div>
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
                            placeholder="Nhập loài"
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
                            placeholder="Nhập giống"
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
                            value={petData.owner.ownerId}
                            onChange={handleChange}
                            placeholder="Nhập mã chủ sở hữu"
                            required
                        />
                    </div>
                    <div className="modal-actions">
                        <button type="submit" className="btn btn-primary">
                            Cập Nhật
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

export default EditPetModal;