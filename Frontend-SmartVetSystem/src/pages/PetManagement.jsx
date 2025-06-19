// PetManagement.jsx
import React, { useEffect, useState } from 'react';
import Sidebar from '../components/Sidebar';
import PetTable from '../components/PetTable';
import Pagination from '../components/Pagination';
import CreatePetModal from '../components/CreatePetModal';
import EditPetModal from '../components/EditPetModal';
import '../styles/dashboard.css';
import { getSidebarGroups } from '../components/sidebarData';
import { fetchAllPets, createPet, updatePet, deletePet } from '../services/petService';
import { useAuth } from '../context/AuthContext';

const PetManagement = () => {
    const [sidebarGroups, setSidebarGroups] = useState([]);
    const [activeSidebarItem, setActiveSidebarItem] = useState('Quản lý thú cưng');
    const [activePage, setActivePage] = useState(1);
    const [searchTerm, setSearchTerm] = useState('');
    const [editingPet, setEditingPet] = useState(null);
    const [petsData, setPetsData] = useState([]);
    const [showCreateModal, setShowCreateModal] = useState(false);

    const { user } = useAuth();

    // Load sidebar groups based on user
    useEffect(() => {
        const groups = getSidebarGroups(user || null);
        setSidebarGroups(groups);
    }, [user]);

    // Load pets data from API
    const loadPets = async () => {
        try {
            const data = await fetchAllPets();
            setPetsData(data);
        } catch (error) {
            console.error('Không thể tải danh sách thú cưng:', error.message);
        }
    };

    useEffect(() => {
        if (activeSidebarItem === 'Quản lý thú cưng') {
            loadPets();
        }
    }, [activeSidebarItem]);

    // Filter pets based on search term
    const filteredPets = petsData.filter(
        (pet) =>
            pet.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            pet.owner?.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    const itemsPerPage = 10;
    const totalPages = Math.ceil(filteredPets.length / itemsPerPage);
    const currentPets = filteredPets.slice(
        (activePage - 1) * itemsPerPage,
        activePage * itemsPerPage
    );

    // Create new pet
    const handleCreatePet = async (petData) => {
        try {
            await createPet(petData);
            await loadPets(); // Reload pet list
            setShowCreateModal(false);
            alert('Tạo thú cưng thành công!');
        } catch (error) {
            console.error('Tạo thú cưng thất bại:', error.message);
            alert('Không thể tạo thú cưng.');
        }
    };

    // Update pet
    const handleUpdatePet = async (petId, updatedData) => {
        try {
            await updatePet(petId, updatedData);
            await loadPets(); // Reload pet list
            setEditingPet(null);
            alert('Cập nhật thú cưng thành công!');
        } catch (error) {
            console.error('Cập nhật thất bại:', error.message);
            alert('Không thể cập nhật thú cưng.');
        }
    };

    // Delete pet
    const handleDeletePet = async (petId) => {
        if (window.confirm('Bạn có chắc chắn muốn xóa?')) {
            try {
                await deletePet(petId);
                await loadPets(); // Reload pet list
                alert('Xóa thú cưng thành công!');
            } catch (error) {
                console.error('Xóa thất bại:', error.message);
                alert('Không thể xóa thú cưng.');
            }
        }
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
            {showCreateModal && (
                <CreatePetModal
                    onClose={() => setShowCreateModal(false)}
                    onCreate={handleCreatePet}
                />
            )}

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
                    <h1>{activeSidebarItem}</h1>
                    <button onClick={() => setShowCreateModal(true)}>Create</button>
                </div>

                {activeSidebarItem === 'Quản lý thú cưng' && (
                    <>
                        <PetTable
                            data={currentPets}
                            onEdit={(pet) => setEditingPet({ ...pet })}
                            onDelete={handleDeletePet}
                        />
                        <Pagination
                            currentPage={activePage}
                            totalPages={totalPages}
                            onChangePage={setActivePage}
                        />
                    </>
                )}

                {editingPet && (
                    <EditPetModal
                        pet={editingPet}
                        onClose={() => setEditingPet(null)}
                        onUpdate={(updatedData) => handleUpdatePet(editingPet.petId, updatedData)}
                    />
                )}
            </div>
        </div>
    );
};

export default PetManagement;