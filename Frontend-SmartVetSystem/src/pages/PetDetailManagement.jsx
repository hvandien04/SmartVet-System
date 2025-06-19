// src/components/PetDetailManagement.jsx
import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import Sidebar from '../components/Sidebar';
import PetDetail from '../components/PetDetail';
import Pagination from '../components/Pagination';
import '../styles/dashboard.css';
import { getSidebarGroups } from '../components/sidebarData';
import { useAuth } from '../context/AuthContext';

const PetDetailManagement = () => {
    const { petId } = useParams();
    const { user } = useAuth();
    const [sidebarGroups, setSidebarGroups] = useState([]);
    const [activeSidebarItem, setActiveSidebarItem] = useState('Quản lý thú cưng');
    const [searchTerm, setSearchTerm] = useState('');
    const [activePage, setActivePage] = useState(1);
    const [totalRecords, setTotalRecords] = useState(0);
    const itemsPerPage = 10;

    // Load sidebar groups
    useEffect(() => {
        const groups = getSidebarGroups(user || null);
        setSidebarGroups(groups);
    }, [user]);

    // Callback để cập nhật số bản ghi
    const handleRecordsUpdate = (count) => {
        setTotalRecords(count);
    };

    return (
        <div style={{ display: 'flex', height: '100vh' }}>
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
                            placeholder="Tìm theo tên thú cưng hoặc chủ sở hữu"
                            value={searchTerm}
                            onChange={(e) => {
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
                    <h1>Chi tiết thú cưng</h1>
                </div>
                <PetDetail
                    petId={petId}
                    activePage={activePage}
                    itemsPerPage={itemsPerPage}
                    onPageChange={setActivePage}
                    onRecordsUpdate={handleRecordsUpdate}
                />
                <Pagination
                    currentPage={activePage}
                    totalPages={Math.ceil(totalRecords / itemsPerPage)}
                    onChangePage={setActivePage}
                />
            </div>
        </div>
    );
};

export default PetDetailManagement;