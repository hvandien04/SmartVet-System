import axios from 'axios';

export const authService = {
    login: async (email, password) => {
        const res = await axios.post('/api/login', { email, password });
        localStorage.setItem('token', res.data.token);
        return true;
    },

    getCurrentUser: async () => {
        const token = localStorage.getItem('token');
        const res = await axios.get('/api/user', {
            headers: { Authorization: `Bearer ${token}` },
        });
        return res.data;
    },

    logout: () => {
        localStorage.removeItem('token');
    }
};
