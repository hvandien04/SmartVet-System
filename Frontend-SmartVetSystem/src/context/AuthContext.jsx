import React, { createContext, useContext, useEffect, useState } from 'react';
import { refreshToken } from '../services/authService';
import { logout as logoutService } from '../services/authService'; // import logout
import {fetchUserInfo as fetchUserInfo} from '../services/authService';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [accessToken, setAccessToken] = useState(null);
    const [loading, setLoading] = useState(true);


    const login = (userData, token) => {
        setUser(userData);
        setAccessToken(token); // lưu vào state (RAM)
    };


    const logout = async () => {
        if (!accessToken) {
            console.warn("Không có accessToken, bỏ qua logout");
            return;
        }

        try {
            await logoutService(accessToken); // truyền token từ context
            setUser(null);
            setAccessToken(null);
        } catch (error) {
            console.error('Logout failed', error);
        }
    };

    useEffect(() => {
        let mounted = true;

        const tryRefreshToken = async () => {
            console.log('Starting token refresh...');
            try {
                const newToken = await refreshToken();
                console.log('New token from refreshToken():', newToken);

                if (!newToken) {
                    throw new Error('Received empty token from refreshToken');
                }

                if (!mounted) return;

                setAccessToken(newToken);

                const userInfo = await fetchUserInfo(newToken);
                if (!mounted) return;

                setUser(userInfo);
                console.log('User set after fetchUserInfo:', userInfo);
            } catch (err) {
                if (!mounted) return;
                console.warn('Token refresh failed:', err.message);
                
                setUser(null);
                setAccessToken(null);
            } finally {
                if (mounted) setLoading(false);
                console.log('Token refresh process finished');
            }
        };

        tryRefreshToken();

        return () => {
            mounted = false;
        };
    }, []);

    if (loading) return <div>Loading...</div>;

    return (
        <AuthContext.Provider
            value={{
                user,
                accessToken,
                login,
                logout,
                isAuthenticated: !!user && !!accessToken,
                loading,
            }}
        >
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => useContext(AuthContext);
