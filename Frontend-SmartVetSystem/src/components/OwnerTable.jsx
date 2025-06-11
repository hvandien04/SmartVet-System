import React from 'react';

const OwnerTable = ({ data, onDelete, onViewProfile }) => (
    <div className="table-container">
        <table>
            <thead>
            <tr>
                <th>No.</th>
                <th>OwnerID</th>
                <th>Full name</th>
                <th>Email</th>
                <th>Phone</th>
                <th>Address</th>
                <th>Profile</th>
                <th>Delete</th>
            </tr>
            </thead>
            <tbody>
            {data.map((doc, index) => (
                <tr key={doc.ownerId}>
                    <td>{index + 1}</td>
                    <td>{doc.ownerId}</td>
                    <td>{doc.name}</td>
                    <td>{doc.email}</td>
                    <td>{doc.phone}</td>
                    <td>{doc.address}</td>
                    <td className="action-icons">
                        <i
                            className="fa fa-address-card"
                            title="View Profile"
                            onClick={() => onViewProfile(doc.ownerId)}
                            style={{ cursor: 'pointer' }}
                        />
                    </td>
                    <td className="action-icons">
                        <i
                            className="fa fa-trash"
                            onClick={() => onDelete(doc.ownerId)}
                            style={{ cursor: 'pointer' }}
                        />
                    </td>
                </tr>
            ))}
            </tbody>
        </table>
    </div>
);

export default OwnerTable;
