import React, { useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';

const Home = () => {
    const { user, logout } = useAuth(); // Thêm loading
    const navigate = useNavigate();

    useEffect(() => {
    }, [user]);

    const handleLogout = async () => {
        await logout();
        navigate('/login');
    };

    const handleGoToLogin = () => {
        navigate('/login');
    };

    return (
        <div style={{ padding: '2rem' }}>
            <h1>Trang chủ</h1>
            {user ? (
                <>
                    <p>Xin chào, {user.fullName || user.email}!</p>
                    <p><strong>ID:</strong> {user.userId}</p>
                    <p><strong>Email:</strong> {user.email}</p>
                    <button onClick={handleLogout}>Đăng xuất</button>
                </>
            ) : (
                <>
                    <p>Bạn chưa đăng nhập.</p>
                    <button onClick={handleGoToLogin}>Quay về trang đăng nhập</button>
                </>
            )}
        </div>
    );
};

export default Home;
