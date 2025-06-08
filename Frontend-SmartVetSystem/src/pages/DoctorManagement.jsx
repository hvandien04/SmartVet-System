import React, { useEffect, useState } from 'react';
import Sidebar from '../components/Sidebar';
import Table from '../components/Table';
import Pagination from '../components/Pagination';
import EditModal from '../components/EditModal';
import '../styles/dashboard.css';
import { fetchAllDoctors } from '../services/doctormanagementService';

const DoctorManagement = () => {
    const sidebarGroups = [
        {
            title: 'Quản trị hệ thống',
            items: [
                'Thống kê',
                'Quản lý bác sĩ',
                'Quản lý thú cưng',
                'Quản lý khách hàng',
            ],
        },
        {
            title: 'Dự đoán bệnh',
            items: [
                'Dự đoán bệnh',
                'Lịch sử dự đoán',
            ],
        },
        {
            title: 'Hồ sơ bệnh án',
            items: [
                'Danh sách bệnh án',
                'Xuất bệnh án',
            ],
        },
        {
            title: 'Lịch khám bệnh',
            items: [
                'Lịch làm việc',
                'Đặt lịch hẹn',
                'Thông báo khách hàng',
            ],
        },
        {
            title: 'User Info',
            isUserInfo: true,
            user: {
                avatarUrl: 'https://i.pravatar.cc/40',
                name: 'Nguyễn Văn A',
            },
        },
    ];
    const [activeSidebarItem, setActiveSidebarItem] = useState('Quản lý bác sĩ');
    const [activePage, setActivePage] = useState(1);
    const [searchTerm, setSearchTerm] = useState('');
    const [editingDoctor, setEditingDoctor] = useState(null);
    const [doctorsData, setDoctorsData] = useState([]);

    useEffect(() => {
        if (activeSidebarItem === 'Quản lý bác sĩ') {
            fetchAllDoctors()
                .then(data => setDoctorsData(data))
                .catch(error => console.error('Không thể tải danh sách bác sĩ:', error));
        }
    }, [activeSidebarItem]);
    const filteredDoctors = doctorsData.filter(doc =>
        doc.fullName.toLowerCase().includes(searchTerm.toLowerCase())
    );

    const itemsPerPage = 3;
    const totalPages = Math.ceil(filteredDoctors.length / itemsPerPage);
    const currentDoctors = filteredDoctors.slice((activePage - 1) * itemsPerPage, activePage * itemsPerPage);

    return (
        <div style={{ display: 'flex', height: '100vh' }}>
            <Sidebar
                groups={sidebarGroups}
                activeItem={activeSidebarItem}
                onItemClick={item => {
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
                            placeholder="Search"
                            value={searchTerm}
                            onChange={e => {
                                setSearchTerm(e.target.value);
                                setActivePage(1);
                            }}
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
                    <button>Create</button>
                </div>

                {activeSidebarItem === 'Quản lý bác sĩ' && (
                    <>
                        <Table
                            data={currentDoctors}
                            onEdit={doc => setEditingDoctor({ ...doc })}
                            onDelete={id =>
                                window.confirm('Are you sure?') &&
                                setDoctorsData(prev => prev.filter(doc => doc.id !== id))
                            }
                        />
                        <Pagination
                            currentPage={activePage}
                            totalPages={totalPages}
                            onChangePage={setActivePage}
                        />
                    </>
                )}

                {editingDoctor && (
                    <EditModal
                        doctor={editingDoctor}
                        onChange={e => {
                            const { name, value } = e.target;
                            setEditingDoctor(prev => ({ ...prev, [name]: value }));
                        }}
                        onSave={() => {
                            setDoctorsData(prev =>
                                prev.map(doc => (doc.id === editingDoctor.id ? editingDoctor : doc))
                            );
                            setEditingDoctor(null);
                        }}
                        onCancel={() => setEditingDoctor(null)}
                    />
                )}
            </div>
        </div>
    );
};

export default DoctorManagement;
