// HistoryTable.jsx
import React from 'react';
import '../styles/dashboard.css';

const HistoryTable = ({ data, onDelete, onViewDetail, onViewMedicalRecord }) => (
    <div className="table-container">
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Diagnosis ID</th>
                    <th>Record ID</th>
                    <th>Pet ID</th>
                    <th>Pet Name</th>
                    <th>Owner</th>
                    <th>Gender</th>
                    <th>Species</th>
                    <th>Breed</th>
                    <th>Detail</th>
                    <th>Delete</th>
                </tr>
            </thead>
            <tbody>
                {data.map((item, index) => (
                    <tr key={item.diagnosisId}>
                        <td>{index + 1}</td>
                        <td>{item.diagnosisId}</td>
                        <td
                            style={{ cursor: 'pointer', color: 'blue', textDecoration: 'underline' }}
                            onClick={() => onViewMedicalRecord(item.recordId)}
                        >
                            {item.recordId}
                        </td>
                        <td>{item.petId}</td>
                        <td>{item.petName}</td>
                        <td>{item.ownerName}</td>
                        <td>{item.gender}</td>
                        <td>{item.species}</td>
                        <td>{item.breed}</td>
                        <td className="action-icons">
                            <i
                                className="fa fa-eye"
                                title="View Diagnosis Detail"
                                onClick={() => onViewDetail(item)}
                                style={{ cursor: 'pointer' }}
                            />
                        </td>
                        <td className="action-icons">
                            <i
                                className="fa fa-trash"
                                title="Delete Medical Record"
                                onClick={() => onDelete(item.recordId)}
                                style={{ cursor: 'pointer' }}
                            />
                        </td>
                    </tr>
                ))}
            </tbody>
        </table>
    </div>
);

export default HistoryTable;