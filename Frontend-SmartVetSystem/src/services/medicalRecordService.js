// src/services/medicalRecordService.js
import api from '../api/axiosConfig';

// Lấy danh sách recordId theo petId (giả định endpoint)
export async function fetchMedicalRecordIdsByPetId(petId) {
    try {
        const response = await api.get(`/pet/${petId}/record-ids`);
        return response.data.result; // Array<string> (danh sách recordId)
    } catch (error) {
        throw new Error('Failed to fetch medical record IDs');
    }
}

// Lấy chi tiết một bản ghi khám bệnh theo recordId
export async function fetchMedicalRecordById(recordId) {
    try {
        const response = await api.get(`/medical-record/${recordId}`);
        return response.data.result; // MedicalRecordResponse
    } catch (error) {
        throw new Error(`Failed to fetch medical record by ID: ${recordId}`);
    }
}