import axios from 'axios';
import { refreshToken } from '../services/authService';

const api = axios.create({
    baseURL: 'http://localhost:8080',
    withCredentials: true,
});

let requestInterceptor;
let responseInterceptor;
let refreshingPromise = null;

let getAccessToken = () => null;

export const attachInterceptors = (getTokenFunc, setAccessToken, logout) => {
    if (requestInterceptor !== undefined) {
        api.interceptors.request.eject(requestInterceptor);
    }
    if (responseInterceptor !== undefined) {
        api.interceptors.response.eject(responseInterceptor);
    }

    getAccessToken = getTokenFunc;

    requestInterceptor = api.interceptors.request.use((config) => {
        const token = getAccessToken();
        if (token) {
            config.headers['Authorization'] = `Bearer ${token}`;
        }
        return config;
    });

    responseInterceptor = api.interceptors.response.use(
        (response) => response,
        async (error) => {
            const originalRequest = error.config;
            if (error.response?.status === 401 && !originalRequest._retry) {
                originalRequest._retry = true;
                try {
                    if (!refreshingPromise) {
                        refreshingPromise = refreshToken()
                            .then((newToken) => {
                                setAccessToken(newToken);
                                return newToken;
                            })
                            .catch((err) => {
                                logout();
                                throw err;
                            })
                            .finally(() => {
                                refreshingPromise = null;
                            });
                    }
                    const newToken = await refreshingPromise;
                    originalRequest.headers['Authorization'] = `Bearer ${newToken}`;
                    return api(originalRequest);
                } catch (refreshError) {
                    return Promise.reject(refreshError);
                }
            }
            return Promise.reject(error);
        }
    );
};

export default api;
