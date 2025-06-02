import axios from 'axios';

export const authService = {
    login: async (email, password) => {
        const res = await axios.post('/api/login', { email, password }, { withCredentials: true });
        return res.data.result.accessToken;  // Cấu trúc tùy backend
    },

    getCurrentUser: async (token) => {
        const res = await axios.get('/api/user', {
            headers: { Authorization: `Bearer ${token}` },
            withCredentials: true,
        });
        return res.data;
    },

    logout: async () => {
        await axios.post('/api/logout', {}, { withCredentials: true });
    },
};
