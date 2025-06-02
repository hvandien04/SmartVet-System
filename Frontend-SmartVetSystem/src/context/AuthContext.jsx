import React, { createContext, useContext, useEffect, useState } from 'react';
import { refreshToken } from '../services/authService';
import { logout as logoutService } from '../services/authService'; // import logout

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [accessToken, setAccessToken] = useState(null);
    const [loading, setLoading] = useState(true);

    async function fetchUserInfo(accessToken) {
        const response = await fetch('http://localhost:8080/user', {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${accessToken}`,
                'Content-Type': 'application/json',
            },
        });

        if (!response.ok) {
            throw new Error('Failed to fetch user info');
        }

        const data = await response.json();
        return data.result; // UserResponse object
    }


    const login = (userData, token) => {
        setUser(userData);
        setAccessToken(token);
        localStorage.setItem('accessToken', token);
    };

    const logout = async () => {
        try {
            await logoutService();
            setUser(null); // xóa user khỏi context
            // Có thể xóa token localStorage nếu bạn lưu
            localStorage.removeItem('access_token');
        } catch (error) {
            console.error('Logout failed', error);
        }
    }


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
                localStorage.setItem('accessToken', newToken);

                const userInfo = await fetchUserInfo(newToken);
                if (!mounted) return;

                setUser(userInfo);
                console.log('User set after fetchUserInfo:', userInfo);
            } catch (err) {
                if (!mounted) return;
                console.warn('Token refresh failed:', err.message);
                setUser(null);
                setAccessToken(null);
                localStorage.removeItem('accessToken');
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
