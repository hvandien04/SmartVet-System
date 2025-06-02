import axios from 'axios';
import { refreshToken } from '../services/authService';

const api = axios.create({
    baseURL: 'http://localhost:8080',
    withCredentials: true
});

// Biến lưu interceptor để eject khi cần
let requestInterceptor;
let responseInterceptor;

// Biến lưu promise để tránh gọi refresh nhiều lần
let refreshingPromise = null;

export const attachInterceptors = (accessToken, setAccessToken, logout) => {
    // Gỡ bỏ các interceptor cũ nếu có
    if (requestInterceptor !== undefined) {
        api.interceptors.request.eject(requestInterceptor);
    }

    if (responseInterceptor !== undefined) {
        api.interceptors.response.eject(responseInterceptor);
    }

    // Interceptor cho request: gắn access token nếu có
    requestInterceptor = api.interceptors.request.use((config) => {
        if (accessToken) {
            config.headers['Authorization'] = `Bearer ${accessToken}`;
        }
        return config;
    });

    // Interceptor cho response: xử lý lỗi 401
    responseInterceptor = api.interceptors.response.use(
        (response) => response,
        async (error) => {
            const originalRequest = error.config;

            // Nếu lỗi 401 và request chưa retry
            if (error.response?.status === 401 && !originalRequest._retry) {
                originalRequest._retry = true;

                try {
                    // Nếu chưa có refresh đang chạy, thì gọi
                    if (!refreshingPromise) {
                        refreshingPromise = refreshToken().then((newToken) => {
                            setAccessToken(newToken);
                            return newToken;
                        }).catch(err => {
                            logout(); // logout nếu refresh thất bại
                            throw err;
                        }).finally(() => {
                            refreshingPromise = null;
                        });
                    }

                    // Đợi token mới
                    const newToken = await refreshingPromise;
                    originalRequest.headers['Authorization'] = `Bearer ${newToken}`;
                    return api(originalRequest); // gửi lại request ban đầu

                } catch (refreshError) {
                    return Promise.reject(refreshError);
                }
            }

            return Promise.reject(error);
        }
    );
};

export default api;
