import React, { useEffect, useState } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import Modal from 'react-modal';
import Sidebar from '../components/Sidebar';
import AppointmentTable from '../components/AppointmentTable';
import CreateAppointment from '../components/CreateAppointment';
import Pagination from '../components/Pagination';
import { getSidebarGroups } from '../components/sidebarData';
import { useAuth } from '../context/AuthContext';
import {
  fetchAllAppointments,
  createAppointment,
  updateAppointment,
  deleteAppointment,
} from '../services/appointmentService';
import '../styles/dashboard.css';

// Đặt root element cho Modal
Modal.setAppElement('#root');

const AppointmentManagement = () => {
  const [sidebarGroups, setSidebarGroups] = useState([]);
  const [activeSidebarItem, setActiveSidebarItem] = useState('Quản lý cuộc hẹn');
  const [activePage, setActivePage] = useState(1);
  const [searchTerm, setSearchTerm] = useState('');
  const [editingAppointment, setEditingAppointment] = useState(null);
  const [appointmentsData, setAppointmentsData] = useState([]);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [loading, setLoading] = useState(false);
  const { user } = useAuth();

  // Tải danh sách cuộc hẹn
  const loadAppointments = async () => {
    try {
      setLoading(true);
      const data = await fetchAllAppointments();
      setAppointmentsData(data);
      setLoading(false);
    } catch (error) {
      console.error('Không thể tải danh sách cuộc hẹn:', error);
      toast.error('Không thể tải danh sách cuộc hẹn.');
      setLoading(false);
    }
  };

  // Cập nhật sidebar dựa trên user
  useEffect(() => {
    const groups = getSidebarGroups(user || null);
    setSidebarGroups(groups);
  }, [user]);

  // Tải dữ liệu khi activeSidebarItem thay đổi
  useEffect(() => {
    if (activeSidebarItem === 'Quản lý cuộc hẹn') {
      loadAppointments();
    }
  }, [activeSidebarItem]);

  // Lọc danh sách cuộc hẹn theo searchTerm
  const filteredAppointments = appointmentsData.filter((appointment) =>
    appointment.appointmentId.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Phân trang
  const itemsPerPage = 10;
  const totalPages = Math.ceil(filteredAppointments.length / itemsPerPage);
  const currentAppointments = filteredAppointments.slice(
    (activePage - 1) * itemsPerPage,
    activePage * itemsPerPage
  );

  // Xử lý tạo cuộc hẹn
  const handleCreateAppointment = async (appointmentData) => {
    try {
      const newAppointment = await createAppointment({
        petId: appointmentData.petId,
        ownerId: appointmentData.ownerId,
        userId: appointmentData.userId,
        appointmentTime: appointmentData.appointmentTime,
        reason: appointmentData.appointmentType,
        content: appointmentData.note,
        status: 'pending',
      });
      setAppointmentsData((prev) => [...prev, newAppointment]);
      setShowCreateModal(false);
      toast.success('Tạo cuộc hẹn thành công!');
    } catch (error) {
      console.error('Tạo cuộc hẹn thất bại:', error);
      toast.error('Không thể tạo cuộc hẹn.');
    }
  };

  // Xử lý cập nhật cuộc hẹn
  const handleUpdateAppointment = async (appointmentId, updateData) => {
    try {
      await updateAppointment(appointmentId, {
        petId: updateData.petId,
        ownerId: updateData.ownerId,
        userId: updateData.userId,
        appointmentTime: updateData.appointmentTime,
        reason: updateData.appointmentType,
        content: updateData.note,
        status: updateData.status || 'pending',
      });
      await loadAppointments();
      setEditingAppointment(null);
      toast.success('Cập nhật cuộc hẹn thành công!');
    } catch (error) {
      console.error('Cập nhật thất bại:', error);
      toast.error('Không thể cập nhật cuộc hẹn.');
    }
  };

  // Xử lý xóa cuộc hẹn
  const handleDeleteAppointment = async (appointmentId) => {
    if (window.confirm('Bạn có chắc chắn muốn xóa?')) {
      try {
        await deleteAppointment(appointmentId);
        await loadAppointments();
        toast.success('Xóa cuộc hẹn thành công!');
      } catch (error) {
        console.error('Xóa thất bại:', error);
        toast.error('Không thể xóa cuộc hẹn.');
      }
    }
  };

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      <ToastContainer position="top-right" autoClose={3000} />
      <Sidebar
        groups={sidebarGroups}
        activeItem={activeSidebarItem}
        onItemClick={(item) => {
          setActiveSidebarItem(item);
          setActivePage(1);
        }}
      />
      <div className="main">
        <div className="top-bar">
          <div className="search-box">
            <i className="fa fa-search search-icon" />
            <input
              type="text"
              placeholder="Tìm kiếm theo Mã cuộc hẹn"
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setActivePage(1);
              }}
            />
            <button onClick={() => setSearchTerm('')}>Xóa</button>
          </div>
          <div className="icons">
            <i className="fa fa-bell" />
            <i className="fa fa-question-circle" />
            <i
              className="fa fa-plus"
              title="Thêm cuộc hẹn"
              onClick={() => setShowCreateModal(true)}
              style={{ cursor: 'pointer', marginLeft: '10px' }}
            />
          </div>
        </div>

        <div className="page-header">
          <h1>{activeSidebarItem}</h1>
          <button onClick={() => setShowCreateModal(true)}>Tạo cuộc hẹn</button>
        </div>

        {loading ? (
          <div>Loading...</div>
        ) : (
          <>
            <AppointmentTable
              data={currentAppointments}
              onDelete={handleDeleteAppointment}
              onViewDetail={(appointment) => setEditingAppointment({ ...appointment })}
            />
            <Pagination
              currentPage={activePage}
              totalPages={totalPages}
              onChangePage={setActivePage}
            />
          </>
        )}

        {showCreateModal && (
          <Modal
            isOpen={showCreateModal}
            onRequestClose={() => setShowCreateModal(false)}
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
            <CreateAppointment
              onSubmit={handleCreateAppointment}
              onCancel={() => setShowCreateModal(false)}
            />
          </Modal>
        )}

        {editingAppointment && (
          <Modal
            isOpen={!!editingAppointment}
            onRequestClose={() => setEditingAppointment(null)}
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
            <CreateAppointment
              onSubmit={(data) => handleUpdateAppointment(editingAppointment.appointmentId, data)}
              onCancel={() => setEditingAppointment(null)}
              initialData={{
                appointmentTime: editingAppointment.appointmentTime,
                petId: editingAppointment.petId,
                ownerId: editingAppointment.owner?.ownerId,
                userId: editingAppointment.user?.userId,
                note: editingAppointment.content,
                appointmentType: editingAppointment.reason,
                status: editingAppointment.status,
              }}
            />
          </Modal>
        )}
      </div>
    </div>
  );
};

export default AppointmentManagement;