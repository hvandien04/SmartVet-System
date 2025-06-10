import api from '../api/axiosConfig';

// Lấy tất cả người dùng (bác sĩ)
export const fetchAllDoctors = async () => {
    try {
        const response = await api.get('/admin');
        return response.data.result;
    } catch (error) {
        console.error('Lỗi khi gọi API /admin:', error);
        throw error;
    }
};

// Tạo mới bác sĩ
export const createDoctor = async (doctorData) => {
    try {
        const response = await api.post('/admin', doctorData);
        return response.data.result;
    } catch (error) {
        console.error('Lỗi khi tạo bác sĩ:', error);
        throw error;
    }
};

// Cập nhật bác sĩ
export const updateDoctor = async (userId, updateData) => {
    try {
        const response = await api.put(`/admin/${userId}`, updateData);
        return response.data.result;
    } catch (error) {
        console.error('Lỗi khi cập nhật bác sĩ:', error);
        throw error;
    }
};

// Xóa bác sĩ (nếu có API)
/*export const deleteDoctor = async (userId) => {
    try {
        const response = await api.delete(`/admin/${userId}`);
        return response.data;
    } catch (error) {
        console.error('Lỗi khi xóa bác sĩ:', error);
        throw error;
    }
};*/
