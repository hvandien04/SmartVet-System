import React from 'react';

const EditModal = ({ doctor, onChange, onSave, onCancel }) => {
    return (
        <div className="modal">
            <div className="modal-content">
                <h3>Edit Doctor</h3>
                <input name="fullName" value={doctor.fullName} onChange={onChange} />
                <input name="email" value={doctor.email} onChange={onChange} />
                <input name="gender" value={doctor.gender} onChange={onChange} />
                <input name="phone" value={doctor.phone} onChange={onChange} />
                <button onClick={onSave}>Save</button>
                <button onClick={onCancel}>Cancel</button>
            </div>
        </div>
    );
};

export default EditModal;
