import React, { useEffect, useState } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { Bar, Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import Sidebar from '../components/Sidebar';
import { useAuth } from '../context/AuthContext';
import { getSidebarGroups } from '../components/sidebarData';
import { fetchAllAppointments } from '../services/appointmentService';
import { fetchAllMedicalRecords } from '../services/medicalRecordManagement';
import '../styles/dashboard.css';

// Đăng ký các thành phần Chart.js
ChartJS.register(CategoryScale, LinearScale, BarElement, LineElement, PointElement, Title, Tooltip, Legend);

const Dashboard = () => {
  const { user, logout } = useAuth();
  const [sidebarGroups, setSidebarGroups] = useState([]);
  const [activeSidebarItem, setActiveSidebarItem] = useState('Dashboard');
  const [appointmentData, setAppointmentData] = useState([]);
  const [medicalRecordData, setMedicalRecordData] = useState([]);
  const [loading, setLoading] = useState(false);

  // Kiểm tra phiên bản trình duyệt để cảnh báo lỗ hổng Microsoft Edge
  useEffect(() => {
    const isEdge = /Edg/.test(navigator.userAgent);
    if (isEdge) {
      const versionMatch = navigator.userAgent.match(/Edg\/(\d+\.\d+)/);
      const version = versionMatch ? parseFloat(versionMatch[1]) : 0;
      if (version < 126) {
        toast.warn('Phiên bản Microsoft Edge của bạn có thể không an toàn. Vui lòng cập nhật lên phiên bản mới nhất.');
      }
    }
  }, []);

  // Tải sidebar dựa trên user
  useEffect(() => {
    const groups = getSidebarGroups(user || null);
    setSidebarGroups(groups);
  }, [user]);

  // Tải dữ liệu cho biểu đồ
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        const [appointments, medicalRecords] = await Promise.all([
          fetchAllAppointments(),
          fetchAllMedicalRecords(),
        ]);
        setAppointmentData(appointments);
        setMedicalRecordData(medicalRecords);
        setLoading(false);
      } catch (error) {
        console.error('Không thể tải dữ liệu dashboard:', error);
        if (error.response?.status === 401) {
          toast.error('Phiên đăng nhập hết hạn, vui lòng đăng nhập lại.');
          logout(); // Đăng xuất nếu token hết hạn
        } else {
          toast.error('Không thể tải dữ liệu dashboard.');
        }
        setLoading(false);
      }
    };
    loadData();
  }, [logout]);

  // Dữ liệu cho biểu đồ cột (trạng thái cuộc hẹn)
  const appointmentStatusCounts = {
    pending: 0,
    completed: 0,
    cancelled: 0,
    no_show: 0,
  };
  appointmentData.forEach((appointment) => {
    const month = new Date(appointment.appointmentTime).getMonth() + 1;
    const currentMonth = new Date().getMonth() + 1;
    if (month === currentMonth) {
      appointmentStatusCounts[appointment.status]++;
    }
  });

  const barChartData = {
    labels: ['Đang chờ', 'Hoàn thành', 'Hủy', 'Không đến'],
    datasets: [
      {
        label: 'Số lượng cuộc hẹn',
        data: [
          appointmentStatusCounts.pending,
          appointmentStatusCounts.completed,
          appointmentStatusCounts.cancelled,
          appointmentStatusCounts.no_show,
        ],
        backgroundColor: [
          'rgba(255, 99, 132, 0.5)',
          'rgba(75, 192, 192, 0.5)',
          'rgba(255, 159, 64, 0.5)',
          'rgba(153, 102, 255, 0.5)',
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(255, 159, 64, 1)',
          'rgba(153, 102, 255, 1)',
        ],
        borderWidth: 1,
      },
    ],
  };

  // Dữ liệu cho biểu đồ đường (hồ sơ khám theo tháng)
  const monthCounts = Array(12).fill(0);
  medicalRecordData.forEach((record) => {
    const year = new Date(record.visit_date).getFullYear();
    const currentYear = new Date().getFullYear();
    if (year === currentYear) {
      const month = new Date(record.visit_date).getMonth();
      monthCounts[month]++;
    }
  });

  const lineChartData = {
    labels: ['Th1', 'Th2', 'Th3', 'Th4', 'Th5', 'Th6', 'Th7', 'Th8', 'Th9', 'Th10', 'Th11', 'Th12'],
    datasets: [
      {
        label: 'Số lượng hồ sơ khám',
        data: monthCounts,
        fill: false,
        borderColor: 'rgba(54, 162, 235, 1)',
        tension: 0.1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: '' }, // Tiêu đề sẽ được đặt trong options của từng biểu đồ
    },
    scales: {
      y: { beginAtZero: true, title: { display: true, text: 'Số lượng' } },
    },
  };

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      <ToastContainer position="top-right" autoClose={3000} />
      <Sidebar
        groups={sidebarGroups}
        activeItem={activeSidebarItem}
        onItemClick={(item) => setActiveSidebarItem(item)}
      />
      <div className="main">
        <div className="page-header">
          <h1>{activeSidebarItem}</h1>
        </div>
        {loading ? (
          <div>Loading...</div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '20px', padding: '20px' }}>
            <div style={{ background: 'white', padding: '20px', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
              <h2>Cuộc Hẹn Theo Trạng Thái (Tháng Hiện Tại)</h2>
              <Bar
                data={barChartData}
                options={{
                  ...chartOptions,
                  plugins: { ...chartOptions.plugins, title: { display: true, text: 'Cuộc Hẹn Theo Trạng Thái' } },
                }}
              />
            </div>
            <div style={{ background: 'white', padding: '20px', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
              <h2>Hồ Sơ Khám Theo Tháng (2025)</h2>
              <Line
                data={lineChartData}
                options={{
                  ...chartOptions,
                  plugins: { ...chartOptions.plugins, title: { display: true, text: 'Hồ Sơ Khám Theo Tháng' } },
                }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;