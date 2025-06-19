import React, { useState, useEffect } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import Modal from 'react-modal';
import Sidebar from '../components/Sidebar';
import ExtractMedicalFile from '../components/ExtractMedicalFile';
import { getSidebarGroups } from '../components/sidebarData';
import { fetchAllMedicalRecords, fetchMedicalRecordById } from '../services/medicalRecordService';
import jsPDF from 'jspdf';
import '../styles/dashboard.css';

Modal.setAppElement('#root');

const ExtractMedicalFileManagement = () => {
  const [sidebarGroups, setSidebarGroups] = useState([]);
  const [activeSidebarItem, setActiveSidebarItem] = useState('Trích xuất bệnh án');
  const [medicalRecords, setMedicalRecords] = useState([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState(null);
  const [recordDetail, setRecordDetail] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setSidebarGroups(getSidebarGroups());
    loadMedicalRecords();
  }, []);

  const loadMedicalRecords = async () => {
    try {
      setLoading(true);
      const result = await fetchAllMedicalRecords();
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
      setLoading(false);
    } catch (error) {
      console.error('Lỗi khi tải hồ sơ y tế:', error.message);
      toast.error('Không thể tải hồ sơ y tế.');
      setLoading(false);
    }
  };

  const fetchRecordDetail = async (recordId) => {
    try {
      setLoading(true);
      const response = await fetchMedicalRecordById(recordId);
      console.log('Fetched record data:', response); // Log dữ liệu từ fetchMedicalRecordById
      // Dữ liệu đã là response.data.result từ fetchMedicalRecordById
      const recordData = response;
      console.log('Processed record detail:', recordData); // Log sau khi xử lý
      if (!recordData || !recordData.recordId) {
        throw new Error('Dữ liệu bệnh án không tồn tại hoặc không hợp lệ');
      }
      setRecordDetail(recordData);
      setLoading(false);
    } catch (error) {
      console.error('Lỗi khi lấy chi tiết bệnh án:', error.message);
      toast.error('Không thể lấy chi tiết bệnh án.');
      setLoading(false);
    }
  };

  const handleExtract = async (record) => {
    setSelectedRecord(record);
    await fetchRecordDetail(record.recordId);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setSelectedRecord(null);
    setRecordDetail(null);
  };

  const generatePDF = (record) => {
    try {
      const doc = new jsPDF();
      doc.setFontSize(16);
      doc.text('HỒ SƠ BỆNH ÁN', 105, 10, { align: 'center' });
      doc.setFontSize(12);
      let y = 20;

      const addSection = (title, content, isArray = false) => {
        doc.text(title, 10, y);
        y += 5;
        if (isArray && content?.length > 0) {
          content.forEach((item, index) => {
            doc.text(`${index + 1}. ${item.imageUrl || 'N/A'} - ${item.description || 'Không có mô tả'}`, 15, y);
            y += 5;
          });
        } else {
          doc.text(content || 'Không có dữ liệu', 15, y);
          y += 5;
        }
        y += 5;
      };

      addSection('Mã bệnh án:', record.recordId || 'N/A');
      addSection('Thú cưng:', `${record.pet?.name || 'N/A'} (ID: ${record.pet?.petId || 'N/A'})`);
      addSection('Loài:', record.pet?.species || 'N/A');
      addSection('Giống:', record.pet?.breed || record.breed || 'N/A');
      addSection('Giới tính:', record.pet?.gender || record.gender || 'N/A');
      addSection('Ngày sinh:', record.pet?.birthDate?.split('T')[0] || 'N/A');
      addSection('Chủ nuôi:', `${record.pet?.owner?.name || 'N/A'} (ID: ${record.pet?.owner?.ownerId || 'N/A'})`);
      addSection('Bác sĩ:', `${record.user?.fullName || 'N/A'} (ID: ${record.user?.userId || 'N/A'})`);
      addSection('Ngày khám:', record.visitDate?.split('T')[0] || 'N/A');
      addSection('Triệu chứng:', record.symptoms || 'N/A');
      addSection('Chẩn đoán:', record.diagnosisSummary || 'N/A');
      addSection('Kết quả xét nghiệm:', record.clinicalTestResults || 'N/A');
      addSection('Loại động vật:', record.animalType || 'N/A');
      addSection('Tuổi (năm):', record.ageYears?.toString() || 'N/A');
      addSection('Cân nặng (kg):', record.weightKg?.toString() || 'N/A');
      addSection('Thời gian bệnh (ngày):', record.durationDays?.toString() || 'N/A');
      addSection('Thể loại thời gian:', record.durationCategory || 'N/A');
      addSection('Mức độ nghiêm trọng:', record.severity || 'N/A');
      addSection('Mùa:', record.season || 'N/A');
      addSection('Khu vực sống:', record.livingArea || 'N/A');
      addSection('Nhiệt độ cơ thể (°C):', record.bodyTemperatureC?.toString() || 'N/A');
      addSection('Nhịp tim (bpm):', record.heartRateBpm?.toString() || 'N/A');
      addSection('Mô tả:', record.description || 'N/A');
      addSection('Kế hoạch điều trị:', record.treatmentPlan || 'N/A');
      addSection('Thuốc kê đơn:', record.medicationsPrescribed || 'N/A');
      addSection('Cần theo dõi:', record.followUpRequired ? 'Có' : 'Không');
      addSection('Ngày tái khám:', record.nextVisitDate?.split('T')[0] || 'N/A');
      addSection('Ghi chú cho chủ nuôi:', record.noteForOwner || 'N/A');
      addSection('Trạng thái:', record.status || 'N/A');
      addSection('Hình ảnh y tế:', record.medicalImages, true);

      doc.save(`MedicalRecord_${record.recordId || 'unknown'}.pdf`);
    } catch (error) {
      console.error('Lỗi khi tạo PDF:', error);
      toast.error('Không thể tạo PDF.');
    }
  };

  const handleConfirmExtract = () => {
    console.log('handleConfirmExtract called', recordDetail);
    if (recordDetail) {
      generatePDF(recordDetail);
      toast.success(`Xuất bệnh án ${recordDetail.recordId} thành công!`);
      closeModal();
    } else {
      toast.error('Chưa tải được dữ liệu bệnh án.');
    }
  };

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      <ToastContainer position="top-right" autoClose={3000} />
      <Sidebar
        groups={sidebarGroups}
        activeItem={activeSidebarItem}
        onItemClick={setActiveSidebarItem}
      />
      <div className="main">
        <div className="top-bar">
          <div className="search-box">
            <i className="fa fa-search search-icon" />
            <input type="text" placeholder="Tìm theo RecordID" />
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

        {loading ? (
          <div>Loading...</div>
        ) : (
          <ExtractMedicalFile data={medicalRecords} onExtract={handleExtract} />
        )}

        <Modal
          isOpen={isModalOpen}
          onRequestClose={closeModal}
          style={{
            content: {
              top: '50%',
              left: '50%',
              right: 'auto',
              bottom: 'auto',
              marginRight: '-50%',
              transform: 'translate(-50%, -50%)',
              width: '500px',
              padding: '20px',
              borderRadius: '8px',
            },
          }}
        >
          <h2>Thông tin bệnh án</h2>
          {loading ? (
            <div>Loading...</div>
          ) : recordDetail ? (
            <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
              <p><strong>Mã bệnh án:</strong> {recordDetail.recordId || 'N/A'}</p>
              <p><strong>Thú cưng:</strong> {recordDetail.pet?.name || 'N/A'} (ID: {recordDetail.pet?.petId || 'N/A'})</p>
              <p><strong>Chủ nuôi:</strong> {recordDetail.pet?.owner?.name || 'N/A'} (ID: {recordDetail.pet?.owner?.ownerId || 'N/A'})</p>
              <p><strong>Bác sĩ:</strong> {recordDetail.user?.fullName || 'N/A'} (ID: {recordDetail.user?.userId || 'N/A'})</p>
              <p><strong>Ngày khám:</strong> {recordDetail.visitDate?.split('T')[0] || 'N/A'}</p>
              <p><strong>Triệu chứng:</strong> {recordDetail.symptoms || 'N/A'}</p>
              <p><strong>Chẩn đoán:</strong> {recordDetail.diagnosisSummary || 'N/A'}</p>
              <p><strong>Kế hoạch điều trị:</strong> {recordDetail.treatmentPlan || 'N/A'}</p>
              <p><strong>Trạng thái:</strong> {recordDetail.status || 'N/A'}</p>
              <p><strong>Hình ảnh y tế:</strong></p>
              {recordDetail.medicalImages?.length > 0 ? (
                <ul>
                  {recordDetail.medicalImages.map((image, index) => (
                    <li key={index}>
                      <a href={image.imageUrl} target="_blank" rel="noopener noreferrer">
                        Hình ảnh {index + 1}
                      </a> - {image.description || 'Không có mô tả'}
                    </li>
                  ))}
                </ul>
              ) : (
                <p>Không có hình ảnh</p>
              )}
            </div>
          ) : (
            <p>Chưa có dữ liệu</p>
          )}
          <div style={{ marginTop: '20px', textAlign: 'right' }}>
            <button
              onClick={handleConfirmExtract}
              style={{
                marginRight: '10px',
                padding: '8px 16px',
                backgroundColor: '#4CAF50',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
              disabled={loading || !recordDetail}
            >
              Xuất PDF
            </button>
            <button
              onClick={closeModal}
              style={{
                padding: '8px 16px',
                backgroundColor: '#f44336',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              Hủy
            </button>
          </div>
        </Modal>
      </div>
    </div>
  );
};

export default ExtractMedicalFileManagement;