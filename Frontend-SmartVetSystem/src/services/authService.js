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

// Hàm login, trả về access token + user info
export const login = async (username, password) => {
    try {
        const response = await api.post('/auth', { username, password });
        const token = response.data.result.token;

        // Sau khi login thành công, gọi fetchUserInfo lấy user info
        const user = await fetchUserInfo();

        return { token, user };
    } catch (error) {
        const message = error.response?.data?.message || 'Login failed';
        throw new Error(message);
    }
};

// Refresh token (axios interceptor cũng có thể tự động gọi hàm này)
export const refreshToken = async () => {
    try {
        const response = await api.post('/auth/refresh-token');
        const token = response.data.result.token;
        if (!token) throw new Error('No token in refresh token response');
        return token;
        // eslint-disable-next-line no-unused-vars
    } catch (error) {
        throw new Error('Failed to refresh token');
    }
};


export async function logout(accessToken) {
    try {
        await api.post('/auth/logout', { token: accessToken });
        return true;
    } catch (error) {
        const message = error.response?.data || 'Logout failed';
        console.error('Logout failed response:', message);
        throw new Error('Logout failed');
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
