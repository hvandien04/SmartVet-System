import api from '../api/axiosConfig';

// Lấy thông tin user hiện tại
export async function fetchUserInfo() {
    try {
        const response = await api.get('/user');
        return response.data.result; // UserResponse object
        // eslint-disable-next-line no-unused-vars
    } catch (error) {
        throw new Error('Failed to fetch user info');
    }
}

// Gửi OTP tới email (forgot password)
export const sendOtpToEmail = async (email) => {
    try {
        await api.post('/user/forgot-password', { email });
        return true;
        // eslint-disable-next-line no-unused-vars
    } catch (error) {
        throw new Error('Failed to send OTP');
    }
};

// Xác thực mã OTP
export const verifyOtpCode = async (email, code) => {
    try {
        const response = await api.post('/user/verify-code', { email, code });
        return response.data.result;
        // eslint-disable-next-line no-unused-vars
    } catch (error) {
        throw new Error('Failed to verify OTP');
    }
};

// Đặt lại mật khẩu
export const resetPassword = async (email, code, newPassword) => {
    try {
        const response = await api.post('/user/reset-password', { email, code, newPassword });
        return response.data.result;
        // eslint-disable-next-line no-unused-vars
    } catch (error) {
        throw new Error('Failed to reset password');
    }
};
