import React, { useState, useEffect } from 'react';
import '../styles/dashboard.css';

const CreateAppointment = ({ onSubmit, onCancel, initialData }) => {
  const [formData, setFormData] = useState({
    petId: '',
    ownerId: '',
    userId: '',
    appointmentTime: '',
    appointmentType: '',
    note: '',
    status: 'pending',
  });

  // Định nghĩa các trạng thái và nhãn tiếng Việt
  const statusOptions = [
    { value: 'pending', label: 'Đang chờ' },
    { value: 'completed', label: 'Hoàn thành' },
    { value: 'cancelled', label: 'Đã hủy' },
    { value: 'no_show', label: 'Không đến' },
  ];

  // Cập nhật formData khi initialData thay đổi (cho chỉnh sửa)
  useEffect(() => {
    if (initialData) {
      const newFormData = {
        petId: initialData.petId || '',
        ownerId: initialData.ownerId || '',
        userId: initialData.userId || '',
        appointmentTime: initialData.appointmentTime
          ? new Date(initialData.appointmentTime).toISOString().slice(0, 16)
          : '',
        appointmentType: initialData.appointmentType || '',
        note: initialData.note || '',
        status: initialData.status || 'pending',
      };
      console.log('Initial formData:', newFormData); // Debug initialData
      setFormData(newFormData);
    }
  }, [initialData]);

  // Xử lý thay đổi input
  const handleChange = (e) => {
    const { name, value } = e.target;
    console.log(`Input changed: ${name} = ${value}`); // Debug thay đổi input
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  // Xử lý submit form
  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('Submitting formData:', formData); // Debug dữ liệu submit
    // Chuyển đổi appointmentTime về định dạng ISO
    const submitData = {
      ...formData,
      appointmentTime: formData.appointmentTime
        ? new Date(formData.appointmentTime).toISOString()
        : null,
    };
    onSubmit(submitData);
  };

  return (
    <div className="form-container">
      <h2>{initialData ? 'Chỉnh sửa cuộc hẹn' : 'Tạo cuộc hẹn mới'}</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Mã thú cưng (Pet ID):</label>
          <input
            type="text"
            name="petId"
            value={formData.petId}
            onChange={handleChange}
            placeholder="Nhập mã thú cưng"
            required
          />
        </div>
        <div className="form-group">
          <label>Mã chủ sở hữu (Owner ID):</label>
          <input
            type="text"
            name="ownerId"
            value={formData.ownerId}
            onChange={handleChange}
            placeholder="Nhập mã chủ sở hữu"
            required
          />
        </div>
        <div className="form-group">
          <label>Mã bác sĩ (User ID):</label>
          <input
            type="text"
            name="userId"
            value={formData.userId}
            onChange={handleChange}
            placeholder="Nhập mã bác sĩ"
            required
          />
        </div>
        <div className="form-group">
          <label>Thời gian hẹn:</label>
          <input
            type="datetime-local"
            name="appointmentTime"
            value={formData.appointmentTime}
            onChange={handleChange}
            required
          />
        </div>
        <div className="form-group">
          <label>Lý do cuộc hẹn:</label>
          <input
            type="text"
            name="appointmentType"
            value={formData.appointmentType}
            onChange={handleChange}
            placeholder="Nhập lý do (ví dụ: Khám định kỳ)"
            required
          />
        </div>
        <div className="form-group">
          <label>Ghi chú:</label>
          <textarea
            name="note"
            value={formData.note}
            onChange={handleChange}
            placeholder="Nhập ghi chú (nội dung cuộc hẹn)"
            rows="4"
          />
        </div>
        <div className="form-group">
          <label>Trạng thái:</label>
          <select
            name="status"
            value={formData.status}
            onChange={handleChange}
            // Loại bỏ disabled để kiểm tra, thêm lại nếu cần
          >
            {statusOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        <div className="form-actions" style={{ marginTop: '20px', textAlign: 'right' }}>
          <button
            type="button"
            onClick={onCancel}
            style={{
              padding: '8px 16px',
              backgroundColor: '#f44336',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              marginRight: '10px',
            }}
          >
            Hủy
          </button>
          <button
            type="submit"
            style={{
              padding: '8px 16px',
              backgroundColor: '#4CAF50',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            {initialData ? 'Cập nhật' : 'Tạo'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default CreateAppointment;