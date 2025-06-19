// PetTable.jsx
import React from 'react';
import { useNavigate } from 'react-router-dom';

const PetTable = ({ data, onEdit, onDelete }) => {
    const navigate = useNavigate();

    const handleViewPetProfile = (pet) => {
        navigate(`/petdetail/${pet.petId}`); // Chuyển hướng đến URL động với petId
    };

    return (
        <div className="table-container">
            <table>
                <thead>
                    <tr>
                        <th>No.</th>
                        <th>Pet ID</th>
                        <th>Name</th>
                        <th>Owner Name</th>
                        <th>Gender</th>
                        <th>Species</th>
                        <th>Breed</th>
                        <th>Birth Date</th>
                        <th>Profile</th>
                        <th>Edit</th>
                        <th>Delete</th>
                    </tr>
                </thead>
                <tbody>
                    {data.map((pet, index) => (
                        <tr key={pet.petId}>
                            <td>{index + 1}</td>
                            <td>{pet.petId}</td>
                            <td>{pet.name}</td>
                            <td>{pet.owner?.name || 'N/A'}</td>
                            <td>{pet.gender}</td>
                            <td>{pet.species}</td>
                            <td>{pet.breed}</td>
                            <td>{pet.birthDate}</td>
                            <td className="action-icons">
                                <i
                                    className="fa fa-clipboard"
                                    onClick={() => handleViewPetProfile(pet)}
                                    style={{ cursor: 'pointer' }}
                                />
                            </td>
                            <td className="action-icons">
                                <i
                                    className="fa fa-pencil"
                                    onClick={() => onEdit(pet)}
                                    style={{ cursor: 'pointer' }}
                                />
                            </td>
                            <td className="action-icons">
                                <i
                                    className="fa fa-trash"
                                    onClick={() => onDelete(pet.petId)}
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

export default PetTable;