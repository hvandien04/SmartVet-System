import React, { createContext, useContext, useEffect, useState } from 'react';
import { refreshToken } from '../services/authService';
import { logout as logoutService } from '../services/authService'; // import logout
import {fetchUserInfo as fetchUserInfo} from '../services/authService';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [accessToken, setAccessToken] = useState(null);


    const login = (userData, token) => {
        setUser(userData);
        setAccessToken(token); // lưu vào state (RAM)
    };


    const logout = async () => {
        if (!accessToken) {
            return;
        }

        try {
            await logoutService(accessToken); // truyền token từ context
            setUser(null);
            setAccessToken(null);
            // eslint-disable-next-line no-unused-vars
        } catch (error) { /* empty */ }
    };

    useEffect(() => {
        let mounted = true;

        const tryRefreshToken = async () => {
            try {
                const newToken = await refreshToken();

                if (!newToken) {
                    throw new Error('Received empty token from refreshToken');
                }

                if (!mounted) return;

                setAccessToken(newToken);

                const userInfo = await fetchUserInfo(newToken);
                if (!mounted) return;

                setUser(userInfo);
                // eslint-disable-next-line no-unused-vars
            } catch (err) {
                if (!mounted) return;

                setUser(null);
                setAccessToken(null);
            } finally { /* empty */ }
        };

        tryRefreshToken();

        return () => {
            mounted = false;
        };
    }, []);

    return (
        <AuthContext.Provider
            value={{
                user,
                accessToken,
                login,
                logout,
                isAuthenticated: !!user && !!accessToken,
            }}
        >
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => useContext(AuthContext);