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
        if (!petId) {
            throw new Error('Invalid or missing petId');
        }
        const response = await api.get(`/medical-record/pet/${petId}`);
        const result = response.data.result || [];
        if (!Array.isArray(result)) {
            console.warn('Expected an array of medical record IDs, received:', result);
            return [];
        }
        // Ánh xạ để chỉ trả về danh sách recordId (nếu API trả về danh sách object)
        return result.map(record => record.recordId || record).filter(id => id);
    } catch (error) {
        const errorMessage = error.response
            ? `Failed to fetch medical record IDs for petId ${petId}, Status: ${error.response.status}, Message: ${error.response.data?.message || 'Unknown error'}`
            : `Failed to fetch medical record IDs for petId ${petId}, Error: ${error.message}`;
        console.error(errorMessage);
        throw new Error(errorMessage);
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
