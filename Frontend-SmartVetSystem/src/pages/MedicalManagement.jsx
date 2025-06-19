// MedicalManagement.jsx
import React, { useEffect, useState } from 'react';
import Sidebar from '../components/Sidebar';
import MedicalRecordTable from '../components/MedicalRecordTable';
import MedicalRecordDetail from '../components/MedicalRecordDetail';
import { getSidebarGroups } from '../components/sidebarData';
import '../styles/dashboard.css';
import { useAuth } from '../context/AuthContext';
import {
    fetchAllMedicalRecords,
    createMedicalRecord,
    updateMedicalRecord,
    deleteMedicalRecord,
} from '../services/medicalRecordService';

const MedicalManagement = () => {
    const { user } = useAuth();
    const [sidebarGroups, setSidebarGroups] = useState([]);
    const [activeSidebarItem, setActiveSidebarItem] = useState('Quản lý hồ sơ y tế');
    const [medicalRecords, setMedicalRecords] = useState([]);
    const [selectedDetail, setSelectedDetail] = useState(null);
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');

    useEffect(() => {
        const groups = getSidebarGroups(user || null);
        setSidebarGroups(groups);
    }, [user]);

    const loadMedicalRecords = async () => {
        try {
            const result = await fetchAllMedicalRecords();
            console.log('Fetched medical records:', result); // Debug dữ liệu từ API
            const formatted = result.map((record) => ({
                recordId: record.recordId,
                petId: record.pet?.petId || 'N/A',
                doctorId: record.user?.userId || 'N/A',
                ownerId: record.pet?.owner?.ownerId || 'N/A',
                detail: record.symptoms || 'N/A',
                diagnose: record.diagnosisSummary || 'N/A',
                treatment: record.treatmentPlan || 'N/A',
                status: record.status || 'N/A',
                fullRecord: record,
            }));
            setMedicalRecords(formatted);
        } catch (error) {
            console.error('Lỗi khi tải hồ sơ y tế:', error.message);
            alert('Không thể tải hồ sơ y tế.');
        }
    };

    useEffect(() => {
        if (activeSidebarItem === 'Quản lý hồ sơ y tế') {
            loadMedicalRecords();
        }
    }, [activeSidebarItem]);

    const handleViewDetail = (item) => {
        console.log('Viewing record:', item.fullRecord); // Debug dữ liệu record
        const record = item.fullRecord;
        setSelectedDetail({
            recordId: record.recordId,
            petId: record.pet?.petId || '',
            userId: record.user?.userId || '',
            visitDate: record.visitDate || '',
            symptoms: record.symptoms || '',
            diagnosisSummary: record.diagnosisSummary || '',
            clinicalTestResults: record.clinicalTestResults || '',
            animalType: record.animalType || '',
            breed: record.breed || '',
            gender: record.gender || '',
            ageYears: record.ageYears || '',
            weightKg: record.weightKg || '',
            durationDays: record.durationDays || '',
            durationCategory: record.durationCategory || '',
            severity: record.severity || '',
            season: record.season || '',
            livingArea: record.livingArea || '',
            bodyTemperatureC: record.bodyTemperatureC || '',
            heartRateBpm: record.heartRateBpm || '',
            description: record.description || '',
            treatmentPlan: record.treatmentPlan || '',
            medicationsPrescribed: record.medicationsPrescribed || '',
            followUpRequired: record.followUpRequired || false,
            nextVisitDate: record.nextVisitDate || '',
            noteForOwner: record.noteForOwner || '',
            status: record.status || '',
            medicalImageRequest: record.medicalImageRequest || [],
        });
    };

    const handleCreateMedicalRecord = async (formData) => {
        try {
            console.log('Creating medical record with data:', formData);
            await createMedicalRecord({
                petId: formData.petId,
                userId: formData.userId,
                visitDate: formData.visitDate,
                symptoms: formData.symptoms || null,
                diagnosisSummary: formData.diagnosisSummary || null,
                clinicalTestResults: formData.clinicalTestResults || null,
                animalType: formData.animalType || null,
                breed: formData.breed || null,
                gender: formData.gender || null,
                ageYears: formData.ageYears,
                weightKg: formData.weightKg,
                durationDays: formData.durationDays,
                durationCategory: formData.durationCategory || null,
                severity: formData.severity || null,
                season: formData.season || null,
                livingArea: formData.livingArea || null,
                bodyTemperatureC: formData.bodyTemperatureC,
                heartRateBpm: formData.heartRateBpm,
                description: formData.description || null,
                treatmentPlan: formData.treatmentPlan || null,
                medicationsPrescribed: formData.medicationsPrescribed || null,
                followUpRequired: formData.followUpRequired,
                nextVisitDate: formData.nextVisitDate || null,
                noteForOwner: formData.noteForOwner || null,
                status: formData.status || null,
                medicalImageRequest: formData.medicalImageRequest || [],
            });
            await loadMedicalRecords();
            setShowCreateModal(false);
            alert('Tạo hồ sơ y tế thành công!');
        } catch (error) {
            console.error('Tạo thất bại:', error.message);
            alert(`Không thể tạo hồ sơ y tế: ${error.message}`);
        }
    };

    const handleSaveDetail = async (formData) => {
        try {
            console.log('Updating medical record with data:', formData);
            await updateMedicalRecord(selectedDetail.recordId, {
                petId: formData.petId,
                userId: formData.userId,
                visitDate: formData.visitDate,
                symptoms: formData.symptoms || null,
                diagnosisSummary: formData.diagnosisSummary || null,
                clinicalTestResults: formData.clinicalTestResults || null,
                animalType: formData.animalType || null,
                breed: formData.breed || null,
                gender: formData.gender || null,
                ageYears: formData.ageYears,
                weightKg: formData.weightKg,
                durationDays: formData.durationDays,
                durationCategory: formData.durationCategory || null,
                severity: formData.severity || null,
                season: formData.season || null,
                livingArea: formData.livingArea || null,
                bodyTemperatureC: formData.bodyTemperatureC,
                heartRateBpm: formData.heartRateBpm,
                description: formData.description || null,
                treatmentPlan: formData.treatmentPlan || null,
                medicationsPrescribed: formData.medicationsPrescribed || null,
                followUpRequired: formData.followUpRequired,
                nextVisitDate: formData.nextVisitDate || null,
                noteForOwner: formData.noteForOwner || null,
                status: formData.status || null,
                medicalImageRequest: formData.medicalImageRequest || [],
            });
            await loadMedicalRecords();
            alert('Cập nhật hồ sơ y tế thành công!');
        } catch (error) {
            console.error('Cập nhật thất bại:', error.message);
            alert(`Không thể cập nhật hồ sơ y tế: ${error.message}`);
        }
    };

    const handleDelete = async (recordId) => {
        if (window.confirm('Bạn có chắc chắn muốn xóa hồ sơ y tế này?')) {
            try {
                await deleteMedicalRecord(recordId);
                await loadMedicalRecords();
                alert('Xóa hồ sơ y tế thành công!');
            } catch (error) {
                console.error('Xóa thất bại:', error.message);
                alert(`Không thể xóa hồ sơ y tế: ${error.message}`);
            }
        }
    };

    const filteredRecords = medicalRecords.filter(
        (item) =>
            item.recordId.toLowerCase().includes(searchTerm.toLowerCase()) ||
            item.petId.toLowerCase().includes(searchTerm.toLowerCase()) ||
            item.doctorId.toLowerCase().includes(searchTerm.toLowerCase()) ||
            item.ownerId.toLowerCase().includes(searchTerm.toLowerCase())
    );

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
                        <input
                            type="text"
                            placeholder="Tìm theo Record ID, Pet ID, Doctor ID, hoặc Owner ID"
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                        />
                        <button>Search</button>
                    </div>
                    <div className="icons">
                        <i className="fa fa-bell" />
                        <i className="fa fa-question-circle" />
                    </div>
                </div>

                <div className="page-header">
                    <h1>{activeSidebarItem}</h1>
                    <button onClick={() => setShowCreateModal(true)}>Tạo hồ sơ y tế</button>
                </div>

                <MedicalRecordTable
                    data={filteredRecords}
                    onViewDetail={handleViewDetail}
                    onDelete={handleDelete}
                />

                <MedicalRecordDetail
                    isOpen={!!selectedDetail}
                    onClose={() => setSelectedDetail(null)}
                    onSave={handleSaveDetail}
                    data={selectedDetail}
                />

                <MedicalRecordDetail
                    isOpen={showCreateModal}
                    onClose={() => setShowCreateModal(false)}
                    onSave={handleCreateMedicalRecord}
                    data={{}}
                />
            </div>
        </div>
    );
};

export default MedicalManagement;