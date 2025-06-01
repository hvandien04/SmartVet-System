import React from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';

const Home = () => {
    const { user, logout } = useAuth();
    const navigate = useNavigate();

    const handleLogout = () => {
        logout(); // Gọi hàm logout từ context
        navigate('/login'); // Điều hướng về trang đăng nhập
    };

    return (
        <div style={{ padding: '2rem' }}>
            <h1>Trang chủ</h1>
            {user ? (
                <>
                    <p>Xin chào, {user.name || user.email}!</p>
                    <button onClick={handleLogout}>Đăng xuất</button>
                </>
            ) : (
                <p>Bạn chưa đăng nhập.</p>
            )}
        </div>
    );
};

export default Home;
