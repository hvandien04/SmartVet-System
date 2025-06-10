import api from '../api/axiosConfig';
import {fetchUserInfo} from './userService';

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

