import React, { useEffect, useState } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import Sidebar from '../components/Sidebar';
import { getSidebarGroups } from '../components/sidebarData';
import { useAuth } from '../context/AuthContext';
import { fetchAppointmentsByTime } from '../services/appointmentService';
import {
  startOfWeek,
  endOfWeek,
  format,
  eachDayOfInterval,
  addDays,
} from 'date-fns';
import DatePicker from 'react-datepicker';
import 'react-toastify/dist/ReactToastify.css';
import 'react-datepicker/dist/react-datepicker.css';
import '../styles/dashboard.css';

const Calendar = () => {
  const { user } = useAuth();
  const [sidebarGroups, setSidebarGroups] = useState([]);
  const [appointments, setAppointments] = useState([]);
  const [activeSidebarItem, setActiveSidebarItem] = useState('Lịch hẹn');
  const [selectedDate, setSelectedDate] = useState(new Date());

  const fromDate = startOfWeek(selectedDate, { weekStartsOn: 1 });
  const toDate = endOfWeek(selectedDate, { weekStartsOn: 1 });

  const daysInWeek = eachDayOfInterval({
    start: fromDate,
    end: toDate,
  });

  useEffect(() => {
    if (user) {
      const groups = getSidebarGroups(user);
      setSidebarGroups(groups);
    }
  }, [user]);

  useEffect(() => {
    if (user) {
      const from = new Date(fromDate.setHours(0, 0, 0, 0)).toISOString();
      const to = new Date(toDate.setHours(23, 59, 59, 999)).toISOString();
      const loadAppointments = async () => {
        try {
          const data = await fetchAppointmentsByTime(from, to);
          setAppointments(data);
        } catch (error) {
          console.error('Không thể tải lịch hẹh:', error);
          toast.error('Không thể tải lịch hẹh.');
        }
      };

      loadAppointments();
    }
  }, [user, selectedDate]);

  const groupAppointmentsByDate = () => {
    const grouped = {};
    daysInWeek.forEach((day) => {
      const dateKey = format(day, 'yyyy-MM-dd');
      grouped[dateKey] = [];
    });

    appointments.forEach((appt) => {
      const dateKey = format(new Date(appt.appointmentTime), 'yyyy-MM-dd');
      if (grouped[dateKey]) {
        grouped[dateKey].push(appt);
      }
    });

    return grouped;
  };

  const groupedAppointments = groupAppointmentsByDate();

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      <ToastContainer position="top-right" autoClose={3000} />
      <Sidebar
        groups={sidebarGroups}
        activeItem={activeSidebarItem}
        onItemClick={(item) => setActiveSidebarItem(item)}
      />
      <div className="main">
        {/* Top header */}
        <div className="top-bar">
          <div className="search-box" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <button onClick={() => setSelectedDate((prev) => addDays(prev, -7))}>
              ← Tuần trước
            </button>

            <DatePicker
              selected={selectedDate}
              onChange={(date) => setSelectedDate(date)}
              dateFormat="dd/MM/yyyy"
              placeholderText="Chọn ngày"
            />

            <button onClick={() => setSelectedDate((prev) => addDays(prev, 7))}>
              Tuần sau →
            </button>
          </div>
          <div className="icons">
            <i className="fa fa-bell" />
            <i className="fa fa-question-circle" />
          </div>
        </div>

        <div className="page-header">
          <h1>
            <i className="fa fa-calendar" style={{ marginRight: '8px' }} />
            Lịch làm việc (tuần: {format(fromDate, 'dd/MM')} - {format(toDate, 'dd/MM')})
          </h1>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: '10px' }}>
          {daysInWeek.map((day) => {
            const dateKey = format(day, 'yyyy-MM-dd');
            const dayLabel = format(day, 'EEE dd/MM');
            return (
              <div
                key={dateKey}
                style={{
                  border: '1px solid #ccc',
                  padding: '10px',
                  borderRadius: '8px',
                  backgroundColor: '#fafafa',
                }}
              >
                <h4 style={{ textAlign: 'center' }}>{dayLabel}</h4>
                {groupedAppointments[dateKey]?.length > 0 ? (
                  groupedAppointments[dateKey].map((appt) => (
                    <div
                      key={appt.appointmentId}
                      style={{
                        marginBottom: '8px',
                        backgroundColor: '#f0f8ff',
                        padding: '6px',
                        borderRadius: '4px',
                        fontSize: '14px',
                      }}
                    >
                      <div><strong>🕒</strong> {format(new Date(appt.appointmentTime), 'HH:mm')}</div>
                      <div><strong>🐶</strong> {appt.pet?.name || 'Không rõ thú cưng'}</div>
                      <div><strong>Lý do:</strong> {appt.reason}</div>
                    </div>
                  ))
                ) : (
                  <div style={{ textAlign: 'center', fontStyle: 'italic', color: '#999' }}>Không có lịch</div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default Calendar;
