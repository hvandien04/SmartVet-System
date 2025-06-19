// src/components/PetDetail.jsx
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/dashboard.css';
import { fetchPetById } from '../services/petService';
import { fetchMedicalRecordIdsByPetId, fetchMedicalRecordById } from '../services/medicalRecordService';

const PetDetail = ({ petId, activePage, itemsPerPage, onPageChange, onRecordsUpdate }) => {
    const navigate = useNavigate();
    const [pet, setPet] = useState(null);
    const [medicalRecords, setMedicalRecords] = useState([]);
    const [loadingPet, setLoadingPet] = useState(true);
    const [loadingRecords, setLoadingRecords] = useState(true);
    const [errorPet, setErrorPet] = useState(null);
    const [errorRecords, setErrorRecords] = useState(null);

    // Debug petId
    console.log('PetDetail: petId from props:', petId);

    // Lấy dữ liệu thú cưng
    useEffect(() => {
        const loadPet = async () => {
            try {
                const data = await fetchPetById(petId);
                console.log('Pet data:', data);
                setPet(data);
                setLoadingPet(false);
            } catch (err) {
                console.error('Error fetching pet:', err);
                setErrorPet(err.message || 'Không thể tải thông tin thú cưng.');
                setLoadingPet(false);
            }
        };
        if (petId) {
            loadPet();
        } else {
            setErrorPet('Không có petId hợp lệ.');
            setLoadingPet(false);
        }
    }, [petId]);

    // Lấy lịch sử khám bệnh
    useEffect(() => {
        const loadMedicalRecords = async () => {
            try {
                // Lấy danh sách recordId
                const recordIds = await fetchMedicalRecordIdsByPetId(petId);
                console.log('Medical record IDs:', recordIds);

                // Lấy chi tiết từng bản ghi
                const records = await Promise.all(
                    recordIds.map(async (recordId) => {
                        const record = await fetchMedicalRecordById(recordId);
                        return record;
                    })
                );

                console.log('Medical records:', records);
                setMedicalRecords(records || []);
                if (onRecordsUpdate) {
                    onRecordsUpdate(records ? records.length : 0); // Cập nhật số bản ghi
                }
                setLoadingRecords(false);
            } catch (err) {
                console.error('Error fetching medical records:', err);
                setErrorRecords(err.message || 'Không thể tải lịch sử khám bệnh.');
                setLoadingRecords(false);
            }
        };
        if (petId) {
            loadMedicalRecords();
        } else {
            setErrorRecords('Không có petId hợp lệ.');
            setLoadingRecords(false);
        }
    }, [petId, onRecordsUpdate]);

    // Định dạng ngày tháng
    const formatDate = (dateString) => {
        try {
            const date = new Date(dateString);
            return date.toLocaleDateString('vi-VN', {
                day: '2-digit',
                month: '2-digit',
                year: 'numeric',
            });
        } catch {
            return dateString || 'N/A';
        }
    };

    // Phân trang lịch sử khám bệnh
    const startIndex = (activePage - 1) * itemsPerPage;
    const currentRecords = medicalRecords.slice(startIndex, startIndex + itemsPerPage);

    if (loadingPet) {
        return <div className="loading">Đang tải thông tin thú cưng...</div>;
    }

    if (errorPet || !pet) {
        return <div className="error">{errorPet || 'Không có dữ liệu thú cưng.'}</div>;
    }

    return (
        <div className="owner-profile">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h2>Thông tin chi tiết / {pet.name}</h2>
                <button className="back-button" onClick={() => navigate('/petmanagement')}>
                    ⟵ Quay lại
                </button>
            </div>

            <div className="profile-section">
                <div className="avatar">
                    <img
                        src="/avatar.png"
                        alt="Avatar"
                        onError={(e) => (e.target.src = '/fallback-avatar.png')}
                    />
                </div>

                <div className="profile-card">
                    <h3>Thông tin thú cưng</h3>
                    <p><strong>Mã thú cưng:</strong> {pet.petId}</p>
                    <p><strong>Tên thú cưng:</strong> {pet.name}</p>
                    <p><strong>Giới tính:</strong> {pet.gender === 'Male' ? 'Đực' : pet.gender === 'Female' ? 'Cái' : pet.gender}</p>
                    <p><strong>Ngày sinh:</strong> {formatDate(pet.birthDate)}</p>
                    <p><strong>Giống:</strong> {pet.breed || 'N/A'}</p>
                    <p><strong>Loài:</strong> {pet.species || 'N/A'}</p>
                    <p><strong>Chủ nuôi:</strong> {pet.owner?.name || 'N/A'}</p>
                    <p><strong>Email chủ nuôi:</strong> {pet.owner?.email || 'N/A'}</p>
                    <p><strong>Số điện thoại:</strong> {pet.owner?.phone || 'N/A'}</p>
                    <p><strong>Địa chỉ:</strong> {pet.owner?.address || 'N/A'}</p>
                </div>
            </div>

            <div className="pet-section">
                <h3>Lịch sử khám bệnh</h3>
                {loadingRecords ? (
                    <div className="loading">Đang tải lịch sử khám bệnh...</div>
                ) : errorRecords ? (
                    <div className="error">{errorRecords}</div>
                ) : medicalRecords.length === 0 ? (
                    <div className="no-data">Chưa có lịch sử khám bệnh.</div>
                ) : (
                    <table className="pet-table">
                        <thead>
                            <tr>
                                <th>Mã khám</th>
                                <th>Ngày khám</th>
                                <th>Chẩn đoán</th>
                                <th>Điều trị</th>
                                <th>Bác sĩ</th>
                            </tr>
                        </thead>
                        <tbody>
                            {currentRecords.map((entry) => (
                                <tr key={entry.id}>
                                    <td>{entry.id}</td>
                                    <td>{formatDate(entry.date)}</td>
                                    <td>{entry.diagnosis || 'N/A'}</td>
                                    <td>{entry.treatment || 'N/A'}</td>
                                    <td>{entry.vet || 'N/A'}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    );
};

export default PetDetail;