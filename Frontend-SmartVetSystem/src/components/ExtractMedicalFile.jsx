import React, { useState } from 'react';
import '../styles/dashboard.css';

const ExtractMedicalFile = ({ data, onExtract }) => {
  const [selectedRecords, setSelectedRecords] = useState([]);

  const handleCheckboxChange = (recordId) => {
    setSelectedRecords((prev) =>
      prev.includes(recordId)
        ? prev.filter((id) => id !== recordId)
        : [...prev, recordId]
    );
  };

  const handleExtract = async (record) => {
    console.log('handleExtract called for record:', record.recordId); // Log để debug
    await onExtract(record, selectedRecords.includes(record.recordId));
  };

  return (
    <div className="table-container">
      <table>
        <thead>
          <tr>
            <th>Check</th>
            <th>RecordID</th>
            <th>Extract</th>
          </tr>
        </thead>
        <tbody>
          {data.map((record) => (
            <tr key={record.recordId}>
              <td>
                <input
                  type="checkbox"
                  checked={selectedRecords.includes(record.recordId)}
                  onChange={() => handleCheckboxChange(record.recordId)}
                />
              </td>
              <td>{record.recordId}</td>
              <td className="action-icons">
                <i
                  className="fa fa-download"
                  title="Extract"
                  onClick={() => handleExtract(record)}
                  style={{ cursor: 'pointer', color: '#4CAF50' }}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ExtractMedicalFile;