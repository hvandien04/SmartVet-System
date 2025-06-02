import React, { createContext, useContext, useEffect, useState } from 'react';
import { refreshToken, introspectToken } from './services/authService'; // introspect là tùy chọn

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [accessToken, setAccessToken] = useState(null);
    const [loading, setLoading] = useState(true);

    const login = (userData, token) => {
        setUser(userData);
        setAccessToken(token);
    };

    const logout = () => {
        setUser(null);
        setAccessToken(null);
        // Optional: gọi API logout hoặc xóa cookie nếu dùng cookie-based
    };

    useEffect(() => {
        const tryRefreshToken = async () => {
            try {
                const newToken = await refreshToken();
                setAccessToken(newToken);

                const userInfo = await introspectToken(newToken);
                setUser(userInfo);
            } catch (err) {
                console.warn('Không thể làm mới token:', err.message);
                logout(); // clear state nếu fail
            } finally {
                setLoading(false);
            }
        };

        tryRefreshToken();
    }, []);

    if (loading) return <div>Loading...</div>;

    return (
        <AuthContext.Provider value={{ user, accessToken, login, logout, setAccessToken }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => useContext(AuthContext);
