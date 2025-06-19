import React from 'react';
import '../styles/dashboard.css';

const AppointmentTable = ({ data, onDelete, onViewDetail }) => {
  // Hàm để định dạng trạng thái với badge
  const getStatusBadge = (status) => {
    const statusStyles = {
      pending: { backgroundColor: '#fff3cd', color: '#856404', label: 'Đang chờ' },
      completed: { backgroundColor: '#d4edda', color: '#155724', label: 'Hoàn thành' },
      cancelled: { backgroundColor: '#f8d7da', color: '#721c24', label: 'Đã hủy' },
      no_show: { backgroundColor: '#d6d8db', color: '#383d41', label: 'Không đến' },
    };

    const style = statusStyles[status] || { backgroundColor: '#e2e3e5', color: '#41464b', label: status };
    return (
      <span
        style={{
          display: 'inline-block',
          padding: '4px 8px',
          borderRadius: '12px',
          fontSize: '12px',
          fontWeight: '500',
          backgroundColor: style.backgroundColor,
          color: style.color,
          textAlign: 'center',
          minWidth: '80px',
        }}
      >
        {style.label}
      </span>
    );
  };

  // Hàm định dạng thời gian
  const formatDateTime = (dateTime) => {
    if (!dateTime) return 'N/A';
    const date = new Date(dateTime);
    return date.toLocaleString('vi-VN', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="table-container">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Mã cuộc hẹn</th>
            <th>Thú cưng</th>
            <th>Chủ sở hữu</th>
            <th>Bác sĩ</th>
            <th>Thời gian hẹn</th>
            <th>Lý do</th>
            <th>Trạng thái</th>
            <th>Chi tiết</th>
            <th>Xóa</th>
          </tr>
        </thead>
        <tbody>
          {data.map((item, index) => (
            <tr key={item.appointmentId}>
              <td>{index + 1}</td>
              <td>{item.appointmentId}</td>
              <td>{item.petId}</td>
              <td>{item.owner?.name || 'N/A'} (ID: {item.owner?.ownerId || 'N/A'})</td>
              <td>{item.user?.fullName || 'N/A'} (ID: {item.user?.userId || 'N/A'})</td>
              <td>{formatDateTime(item.appointmentTime)}</td>
              <td>{item.reason}</td>
              <td>{getStatusBadge(item.status)}</td>
              <td className="action-icons">
                <i
                  className="fa fa-eye"
                  title="Xem chi tiết"
                  onClick={() => onViewDetail(item)}
                  style={{ cursor: 'pointer' }}
                />
              </td>
              <td className="action-icons">
                <i
                  className="fa fa-trash"
                  title="Xóa"
                  onClick={() => onDelete(item.appointmentId)}
                  style={{ cursor: 'pointer' }}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default AppointmentTable;