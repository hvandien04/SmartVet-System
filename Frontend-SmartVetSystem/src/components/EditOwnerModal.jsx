import React, { useState, useEffect } from 'react';
import '../styles/dashboard.css';

const EditOwnerModal = ({ owner, onClose, onUpdate }) => {
    const [ownerData, setOwnerData] = useState({
        name: '',
        email: '',
        phone: '',
        address: '',
    });

    useEffect(() => {
        if (owner) {
            setOwnerData({
                name: owner.name || '',
                email: owner.email || '',
                phone: owner.phone || '',
                address: owner.address || '',
            });
        }
    }, [owner]);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setOwnerData(prev => ({
            ...prev,
            [name]: value,
        }));
    };

    const handleSubmit = () => {
        onUpdate(owner.ownerId, ownerData);
        onClose();
    };

    return (
        <div className="modal-overlay">
            <div className="modal">
                <h2>✏️ Chỉnh sửa thông tin khách hàng</h2>

                <div className="form-group">
                    <label>Họ và tên</label>
                    <input
                        name="name"
                        value={ownerData.name}
                        onChange={handleChange}
                        placeholder="Nhập họ tên khách hàng"
                    />
                </div>

                <div className="form-group">
                    <label>Email</label>
                    <input
                        name="email"
                        value={ownerData.email}
                        onChange={handleChange}
                        placeholder="Nhập email khách hàng"
                    />
                </div>

                <div className="form-group">
                    <label>Số điện thoại</label>
                    <input
                        name="phone"
                        value={ownerData.phone}
                        onChange={handleChange}
                        placeholder="Nhập số điện thoại khách hàng"
                    />
                </div>

                <div className="form-group">
                    <label>Địa chỉ</label>
                    <input
                        name="address"
                        value={ownerData.address}
                        onChange={handleChange}
                        placeholder="Nhập địa chỉ khách hàng"
                    />
                </div>

                <div style={{ marginTop: '20px', textAlign: 'right' }}>
                    <button onClick={onClose}>Hủy</button>
                    <button onClick={handleSubmit}>Cập nhật</button>
                </div>
            </div>
        </div>
    );
};

export default EditOwnerModal;
