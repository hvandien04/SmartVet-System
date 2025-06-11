import React, { useEffect, useState } from 'react';
import Sidebar from '../components/Sidebar';
import Table from '../components/OwnerTable.jsx';
import Pagination from '../components/Pagination';
import '../styles/dashboard.css';
import SearchBox from '../components/SearchBox';
import { fetchAllOwners, createOwner, deleteOwner, fetchOwnerById } from '../services/ownerService';
import CreateOwnerModal from '../components/CreateOwnerModal';
import { getSidebarGroups } from '../components/sidebarData';
import OwnerProfile from '../components/OwnerProfile';
import { useAuth } from '../context/AuthContext';

const OwnerManagement = () => {
    const { user } = useAuth();

    const [sidebarGroups, setSidebarGroups] = useState([]);
    const [activeSidebarItem, setActiveSidebarItem] = useState('Quản lý khách hàng');
    const [activePage, setActivePage] = useState(1);
    const [searchTerm, setSearchTerm] = useState('');
    const [OwnersData, setOwnersData] = useState([]);
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [selectedOwner, setSelectedOwner] = useState(null);

    useEffect(() => {
        const groups = getSidebarGroups(user || null);
        setSidebarGroups(groups);
    }, [user]);


    const loadOwners = async () => {
        try {
            const data = await fetchAllOwners();
            setOwnersData(data);
        } catch (error) {
            console.error('Không thể tải danh sách khách hàng:', error);
        }
    };

    useEffect(() => {
        if (activeSidebarItem === 'Quản lý khách hàng') {
            loadOwners();
        }
    }, [activeSidebarItem]);

    const handleViewProfile = async (ownerId) => {
        try {
            const detailedOwner = await fetchOwnerById(ownerId);
            setSelectedOwner(detailedOwner);
        } catch (error) {
            console.error('Không thể tải chi tiết khách hàng:', error);
            alert('Lỗi khi tải thông tin khách hàng.');
        }
    };

    const handleBack = () => {
        setSelectedOwner(null);
    };

    const filteredOwners = OwnersData.filter(doc =>
        doc.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    const itemsPerPage = 10;
    const totalPages = Math.ceil(filteredOwners.length / itemsPerPage);
    const currentOwners = filteredOwners.slice((activePage - 1) * itemsPerPage, activePage * itemsPerPage);

    const handleCreateOwner = async (OwnerData) => {
        try {
            await createOwner(OwnerData);
            await loadOwners();
            setShowCreateModal(false);
        } catch (error) {
            console.error('Tạo khách hàng thất bại:', error);
            alert('Không thể tạo khách hàng.');
        }
    };

    return (
        <div style={{ display: 'flex', height: '100vh' }}>
            {showCreateModal && (
                <CreateOwnerModal
                    onClose={() => setShowCreateModal(false)}
                    onCreate={handleCreateOwner}
                />
            )}

            <Sidebar
                groups={sidebarGroups}
                activeItem={activeSidebarItem}
                onItemClick={item => {
                    setActiveSidebarItem(item);
                    setActivePage(1);
                }}
            />

            <div className="main">
                <SearchBox
                    searchTerm={searchTerm}
                    setSearchTerm={(value) => {
                        setSearchTerm(value);
                        setActivePage(1);
                    }}
                />

                <div className="page-header">
                    <h1>{activeSidebarItem}</h1>
                    <button onClick={() => setShowCreateModal(true)}>Create</button>
                </div>

                {activeSidebarItem === 'Quản lý khách hàng' && (
                    <>
                        {selectedOwner ? (
                            <OwnerProfile owner={selectedOwner} onBack={handleBack} />
                        ) : (
                            <>
                                <Table
                                    data={currentOwners}
                                    onDelete={async (id) => {
                                        if (window.confirm('Bạn có chắc chắn muốn xóa khách hàng này?')) {
                                            try {
                                                await deleteOwner(id);
                                                await loadOwners();
                                            } catch (error) {
                                                console.error('Xóa khách hàng thất bại:', error);
                                                alert('Không thể xóa khách hàng.');
                                            }
                                        }
                                    }}
                                    onViewProfile={handleViewProfile}
                                />
                                <Pagination
                                    currentPage={activePage}
                                    totalPages={totalPages}
                                    onChangePage={setActivePage}
                                />
                            </>
                        )}
                    </>
                )}
            </div>
        </div>
    );
};

export default OwnerManagement;
