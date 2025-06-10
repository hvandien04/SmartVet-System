import React from 'react';

const Table = ({ data, onEdit, onDelete }) => (
    <div className="table-container">
        <table>
            <thead>
            <tr>
                <th>No.</th>
                <th>DoctorID</th>
                <th>Full name</th>
                <th>Email</th>
                <th>Address</th>
                <th>Phone</th>
                <th>Role</th>
                <th>Edit</th>
                <th>Delete</th>
            </tr>
            </thead>
            <tbody>
            {data.map((doc, index) => (
                <tr key={doc.userId}>
                    <td>{index + 1}</td>
                    <td>{doc.userId}</td>
                    <td>{doc.fullName}</td>
                    <td>{doc.email}</td>
                    <td>{doc.address}</td>
                    <td>{doc.phone}</td>
                    <td>{doc.role}</td>
                    <td className="action-icons">
                        <i className="fa fa-pencil" onClick={() => onEdit(doc)} />
                    </td>
                    <td className="action-icons">
                        <i className="fa fa-trash" onClick={() => onDelete(doc.id)} />
                    </td>
                </tr>
            ))}
            </tbody>
        </table>
    </div>
);

export default Table;
