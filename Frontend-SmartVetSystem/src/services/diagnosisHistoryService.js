// diagnosisHistoryService.js
import api from '../api/axiosConfig';

// Lấy tất cả Diagnosis History
export async function fetchAllDiagnosisHistories() {
    try {
        const response = await api.get('/diagnosis-history');
        return response.data.result; // List<DiagnosisHistoryResponse>
    } catch (error) {
        console.error('Failed to fetch diagnosis histories:', error.response?.data || error.message);
        throw new Error(error.response?.data?.message || 'Failed to fetch diagnosis histories');
    }
}