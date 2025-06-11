import React, { useEffect, useState } from "react";
import '../styles/dashboard.css';
import EditOwnerModal from '../components/EditOwnerModal';
import { updateOwner } from '../services/ownerService';
import { fetchPetsByOwner } from '../services/petService';
import AddPetModal from '../components/AddPetByOwnerIDModal';
import { createPet } from '../services/petService';

const OwnerProfile = ({ owner, onBack }) => {
    const [showEditModal, setShowEditModal] = useState(false);
    const [ownerData, setOwnerData] = useState(owner);
    const [pets, setPets] = useState([]);
    const [showAddPetModal, setShowAddPetModal] = useState(false);

    const handleCreatePet = async (newPet) => {
        try {
            await createPet(newPet);
            const updatedPets = await fetchPetsByOwner(owner.ownerId);
            setPets(updatedPets); // làm mới danh sách
        } catch (error) {
            alert('Thêm thú cưng thất bại!');
            console.error(error);
        }
    };


    useEffect(() => {
        const loadPets = async () => {
            try {
                if (owner.ownerId) {
                    const petList = await fetchPetsByOwner(owner.ownerId);
                    setPets(petList);
                }
            } catch (error) {
                console.error("Failed to fetch pets:", error);
            }
        };
        loadPets();
    }, [owner.ownerId]); // Gọi lại khi ownerId thay đổi

    const handleUpdate = async (ownerId, newData) => {
        try {
            const updated = await updateOwner(ownerId, newData);
            setOwnerData(prev => ({
                ...prev,
                ...updated
            }));
        } catch (error) {
            alert('Cập nhật thất bại!');
            console.error(error);
        }
    };

    return (
        <div className="owner-profile">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h2>Thông tin chi tiết / {ownerData.name}</h2>
                <button className="back-button" onClick={onBack}>⟵ Quay lại danh sách</button>
            </div>

            <div className="profile-section">
                <div className="avatar">
                    <img src="/avatar.png" alt="Avatar" />
                </div>

                <div className="profile-card">
                    <h3>Profile</h3>
                    <p><strong>Mã chủ nuôi:</strong> {ownerData.ownerId}</p>
                    <p><strong>Tên đầy đủ:</strong> {ownerData.name}</p>
                    <p><strong>Email:</strong> {ownerData.email}</p>
                    <p><strong>Số điện thoại:</strong> {ownerData.phone}</p>
                    <p><strong>Địa chỉ:</strong> {ownerData.address}</p>
                    <button className="edit-button" onClick={() => setShowEditModal(true)}>Edit</button>
                </div>
            </div>

            <div className="pet-section">
                <button className="add-button" onClick={() => setShowAddPetModal(true)}>Add</button>
                <table className="pet-table">
                    <thead>
                    <tr>
                        <th>PetID</th>
                        <th>Name</th>
                        <th>Birthday</th>
                        <th>Breed</th>
                        <th>Species</th>
                    </tr>
                    </thead>
                    <tbody>
                    {pets.length > 0 ? (
                        pets.map((pet, index) => (
                            <tr key={index}>
                                <td>{pet.petId}</td>
                                <td>{pet.name}</td>
                                <td>{pet.birthDate}</td>
                                <td>{pet.breed}</td>
                                <td>{pet.species}</td>
                            </tr>
                        ))
                    ) : (
                        <tr>
                            <td colSpan="5">Không có thú cưng nào.</td>
                        </tr>
                    )}
                    </tbody>
                </table>
            </div>

            {showAddPetModal && (
                <AddPetModal
                    ownerId={owner.ownerId}
                    onClose={() => setShowAddPetModal(false)}
                    onCreate={handleCreatePet}
                />
            )}


            {showEditModal && (
                <EditOwnerModal
                    owner={ownerData}
                    onClose={() => setShowEditModal(false)}
                    onUpdate={handleUpdate}
                />
            )}
        </div>
    );
};

export default OwnerProfile;
