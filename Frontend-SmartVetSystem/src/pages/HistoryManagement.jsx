// HistoryManagement.jsx
import React, { useEffect, useState } from 'react';
import Sidebar from '../components/Sidebar';
import HistoryTable from '../components/HistoryTable';
import HistoryPredictDetail from '../components/HistoryDetail';
import { getSidebarGroups } from '../components/sidebarData';
import '../styles/dashboard.css';
import { useAuth } from '../context/AuthContext';
import { fetchAllDiagnosisHistories } from '../services/diagnosisHistoryService';
import { fetchMedicalRecordById, deleteMedicalRecord } from '../services/medicalRecordService';

const HistoryManagement = () => {
    const { user } = useAuth();
    const [sidebarGroups, setSidebarGroups] = useState([]);
    const [activeSidebarItem, setActiveSidebarItem] = useState('Lịch sử dự đoán');
    const [histories, setHistories] = useState([]);
    const [selectedDetail, setSelectedDetail] = useState(null);
    const [selectedMedicalRecord, setSelectedMedicalRecord] = useState(null);

    useEffect(() => {
        const groups = getSidebarGroups(user || null);
        setSidebarGroups(groups);
    }, [user]);

    useEffect(() => {
        const loadHistories = async () => {
            try {
                const result = await fetchAllDiagnosisHistories();
                const formatted = result.map((history) => ({
                    diagnosisId: history.diagnosisId,
                    recordId: history.medicalRecord.recordId,
                    petId: history.medicalRecord.pet.petId,
                    petName: history.medicalRecord.pet.name,
                    ownerName: history.medicalRecord.pet.owner.name,
                    gender: history.medicalRecord.pet.gender,
                    species: history.medicalRecord.pet.species,
                    breed: history.medicalRecord.pet.breed,
                    createdAt: history.createdAt.split('T')[0],
                    // Lưu toàn bộ medicalRecord để sử dụng nếu cần
                    medicalRecord: history.medicalRecord,
                }));
                setHistories(formatted);
            } catch (error) {
                console.error('Lỗi khi tải lịch sử chẩn đoán:', error.message);
                alert('Không thể tải lịch sử chẩn đoán.');
            }
        };

        loadHistories();
    }, []);

    const handleViewMedicalRecord = async (recordId) => {
        try {
            const medicalRecord = await fetchMedicalRecordById(recordId);
            setSelectedMedicalRecord(medicalRecord);
        } catch (error) {
            console.error('Lỗi khi tải chi tiết hồ sơ y tế:', error.message);
            alert('Không thể tải chi tiết hồ sơ y tế.');
        }
    };

    const handleDelete = async (recordId) => {
        if (window.confirm('Bạn có chắc chắn muốn xóa hồ sơ y tế này?')) {
            try {
                await deleteMedicalRecord(recordId);
                // Tải lại danh sách Diagnosis History
                const result = await fetchAllDiagnosisHistories();
                const formatted = result.map((history) => ({
                    diagnosisId: history.diagnosisId,
                    recordId: history.medicalRecord.recordId,
                    petId: history.medicalRecord.pet.petId,
                    petName: history.medicalRecord.pet.name,
                    ownerName: history.medicalRecord.pet.owner.name,
                    gender: history.medicalRecord.pet.gender,
                    species: history.medicalRecord.pet.species,
                    breed: history.medicalRecord.pet.breed,
                    createdAt: history.createdAt.split('T')[0],
                    medicalRecord: history.medicalRecord,
                }));
                setHistories(formatted);
                alert('Xóa hồ sơ y tế thành công!');
            } catch (error) {
                console.error('Xóa thất bại:', error.message);
                alert(`Không thể xóa hồ sơ y tế: ${error.message}`);
            }
        }
    };

    const handleCopyProfile = (item) => {
        navigator.clipboard.writeText(JSON.stringify(item));
        alert('Đã sao chép hồ sơ!');
    };

    return (
        <div style={{ display: 'flex', height: '100vh' }}>
            <Sidebar
                groups={sidebarGroups}
                activeItem={activeSidebarItem}
                onItemClick={setActiveSidebarItem}
            />
            <div className="main">
                <div className="top-bar">
                    <div className="search-box">
                        <i className="fa fa-search search-icon" />
                        <input type="text" placeholder="Search by name or owner" />
                        <button>Search</button>
                    </div>
                    <div className="icons">
                        <i className="fa fa-bell" />
                        <i className="fa fa-question-circle" />
                    </div>
                </div>

                <div className="page-header">
                    <h1>{activeSidebarItem}</h1>
                </div>

                <HistoryTable
                    data={histories}
                    onDelete={handleDelete}
                    onCopyProfile={handleCopyProfile}
                    onViewDetail={setSelectedDetail}
                    onViewMedicalRecord={handleViewMedicalRecord}
                />

                <HistoryPredictDetail
                    isOpen={!!selectedDetail}
                    onClose={() => setSelectedDetail(null)}
                    data={selectedDetail}
                />

                {selectedMedicalRecord && (
                    <HistoryPredictDetail
                        isOpen={!!selectedMedicalRecord}
                        onClose={() => setSelectedMedicalRecord(null)}
                        data={{
                            recordId: selectedMedicalRecord.recordId,
                            petName: selectedMedicalRecord.pet?.name,
                            ownerName: selectedMedicalRecord.pet?.owner?.name,
                            visitDate: selectedMedicalRecord.visitDate,
                            symptoms: selectedMedicalRecord.symptoms,
                            diagnosisSummary: selectedMedicalRecord.diagnosisSummary,
                            treatmentPlan: selectedMedicalRecord.treatmentPlan,
                            medicationsPrescribed: selectedMedicalRecord.medicationsPrescribed,
                            // Thêm các trường khác nếu cần
                        }}
                    />
                )}
            </div>
        </div>
    );
};

export default HistoryManagement;