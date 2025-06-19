import api from '../api/axiosConfig';

// GET: Lấy tất cả hồ sơ y tế
export async function fetchAllMedicalRecords() {
    try {
        const response = await api.get('/medical-record');
        return response.data.result;
    } catch (error) {
        throw new Error('Failed to fetch medical records');
    }
}

// GET: Lấy chi tiết hồ sơ y tế theo recordId
export async function fetchMedicalRecordById(recordId) {
    try {
        const response = await api.get(`/medical-record/${recordId}`);
        return response.data.result;
    } catch (error) {
        throw new Error(`Failed to fetch medical record by ID: ${recordId}`);
    }
}

// GET: Lấy danh sách recordId theo petId
export async function fetchMedicalRecordIdsByPetId(petId) {
    try {
        const response = await api.get(`/pet/${petId}/record-ids`);
        return response.data.result;
    } catch (error) {
        throw new Error('Failed to fetch medical record IDs');
    }
}

// POST: Tạo mới hồ sơ y tế
export async function createMedicalRecord(data) {
    try {
        const response = await api.post('/medical-record', data);
        return response.data.result;
    } catch (error) {
        throw new Error('Failed to create medical record');
    }
}

// PUT: Cập nhật hồ sơ y tế
export async function updateMedicalRecord(recordId, data) {
    try {
        const response = await api.put(`/medical-record/${recordId}`, data);
        return response.data.result;
    } catch (error) {
        const errorMessage = error.response?.data?.message || `Failed to update medical record: ${recordId}`;
        console.error('Update medical record error:', {
            status: error.response?.status,
            data: error.response?.data,
            message: errorMessage,
        });
        throw new Error(errorMessage);
    }
}

// DELETE: Xóa hồ sơ y tế
export async function deleteMedicalRecord(recordId) {
    try {
        const response = await api.delete(`/medical-record/${recordId}`);
        return response.data.result;
    } catch (error) {
        throw new Error('Failed to delete medical record');
    }
}
