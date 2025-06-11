import React, { useEffect, useState } from 'react';
import Sidebar from '../components/Sidebar';
import Table from '../components/DoctorTable.jsx';
import Pagination from '../components/Pagination';
import '../styles/dashboard.css';
import { fetchAllDoctors } from '../services/adminService.js';
import CreateDoctorModal from '../components/CreateDoctorModal';
import { createDoctor } from '../services/adminService.js';
import EditDoctorModal from '../components/EditDoctorModal';
import { updateDoctor } from '../services/adminService.js';
import { getSidebarGroups } from '../components/sidebarData';
import { useAuth } from '../context/AuthContext';

const DoctorManagement = () => {

    const [sidebarGroups, setSidebarGroups] = useState([]);
    const [activeSidebarItem, setActiveSidebarItem] = useState('Quản lý bác sĩ');
    const [activePage, setActivePage] = useState(1);
    const [searchTerm, setSearchTerm] = useState('');
    const [editingDoctor, setEditingDoctor] = useState(null);
    const [doctorsData, setDoctorsData] = useState([]);
    const [showCreateModal, setShowCreateModal] = useState(false);

    const loadDoctors = async () => {
        try {
            const data = await fetchAllDoctors();
            setDoctorsData(data);
        } catch (error) {
            console.error('Không thể tải danh sách bác sĩ:', error);
        }
    };

    const { user } = useAuth();

    useEffect(() => {
        const groups = getSidebarGroups(user || null);
        setSidebarGroups(groups);
    }, [user]);

    useEffect(() => {
        if (activeSidebarItem === 'Quản lý bác sĩ') {
            loadDoctors();
        }
    }, [activeSidebarItem]);

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

    const itemsPerPage = 10;
    const totalPages = Math.ceil(filteredDoctors.length / itemsPerPage);
    const currentDoctors = filteredDoctors.slice((activePage - 1) * itemsPerPage, activePage * itemsPerPage);
    const handleCreateDoctor = async (doctorData) => {
        try {
            await createDoctor(doctorData);
            await loadDoctors(); // tải lại danh sách
            setShowCreateModal(false);
        } catch (error) {
            console.error('Tạo bác sĩ thất bại:', error);
            alert('Không thể tạo bác sĩ.');
        }
    };


    const handleUpdateDoctor = async (userId, updateData) => {
        try {
            await updateDoctor(userId, updateData);
            await loadDoctors(); // 🔁 tải lại danh sách
            setEditingDoctor(null); // đóng modal nếu cần
        } catch (error) {
            console.error('Cập nhật thất bại:', error);
            alert('Không thể cập nhật bác sĩ.');
        }
    };


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
            {showCreateModal && (
                <CreateDoctorModal
                    onClose={() => setShowCreateModal(false)}
                    onCreate={handleCreateDoctor}
                />
            )}

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
                    <button onClick={() => setShowCreateModal(true)}>Create</button>
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
                    <EditDoctorModal
                        doctor={editingDoctor}
                        onClose={() => setEditingDoctor(null)}
                        onUpdate={handleUpdateDoctor}
                    />
                )}

            </div>
        </div>
    );
};

export default DoctorManagement;
