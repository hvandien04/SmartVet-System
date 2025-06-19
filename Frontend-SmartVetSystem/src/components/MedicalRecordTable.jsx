// MedicalRecordTable.jsx
import React from 'react';
import '../styles/dashboard.css';

const MedicalRecordTable = ({ data, onViewDetail, onDelete }) => (
    <div className="table-container">
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Record ID</th>
                    <th>Pet ID</th>
                    <th>Doctor ID</th>
                    <th>Owner ID</th>
                    <th>Detail</th>
                    <th>Diagnose</th>
                    <th>Treatment</th>
                    <th>Status</th>
                    <th>View</th>
                    <th>Delete</th>
                </tr>
            </thead>
            <tbody>
                {data.map((item, index) => (
                    <tr key={item.recordId}>
                        <td>{index + 1}</td>
                        <td>{item.recordId}</td>
                        <td>{item.petId}</td>
                        <td>{item.doctorId}</td>
                        <td>{item.ownerId}</td>
                        <td>{item.detail}</td>
                        <td>{item.diagnose}</td>
                        <td>{item.treatment}</td>
                        <td>{item.status}</td>
                        <td className="action-icons">
                            <i
                                className="fa fa-eye"
                                title="View Detail"
                                onClick={() => onViewDetail(item)}
                                style={{ cursor: 'pointer' }}
                            />
                        </td>
                        <td className="action-icons">
                            <i
                                className="fa fa-trash"
                                title="Delete"
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

export default MedicalRecordTable;