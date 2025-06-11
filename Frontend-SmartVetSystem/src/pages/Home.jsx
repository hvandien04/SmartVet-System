import React, { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import Sidebar from '../components/Sidebar';
import { getSidebarGroups } from '../components/sidebarData';
import '../styles/dashboard.css';
import SearchBox from '../components/SearchBox';

const Home = () => {
    const { user } = useAuth();
    const navigate = useNavigate();
    const [searchTerm, setSearchTerm] = useState('');

    const [sidebarGroups, setSidebarGroups] = useState([]);
    const [activeSidebarItem, setActiveSidebarItem] = useState('');

    useEffect(() => {
        // Dù có user hay không, vẫn lấy sidebar phù hợp (nếu có cần phân quyền)
        const groups = getSidebarGroups(user || {});
        setSidebarGroups(groups);
    }, [user]);

    const handleGoToLogin = () => {
        navigate('/login');
    };

    return (
        <div style={{ display: 'flex', height: '100vh' }}>
            <Sidebar
                groups={sidebarGroups}
                activeItem={activeSidebarItem}
                onItemClick={item => {
                    setActiveSidebarItem(item);
                }}
            />

            <SearchBox
                searchTerm={searchTerm}
                setSearchTerm={(value) => {
                    setSearchTerm(value);
                }}
            />


            <div className="main">
                <div className="page-header">
                    <h1>{activeSidebarItem}</h1>
                </div>

                {/* Nếu chưa đăng nhập, hiển thị nút quay về đăng nhập */}
                {!user && (
                    <div style={{ padding: '1rem' }}>
                        <p>Bạn chưa đăng nhập.</p>
                        <button className="back-to-login-btn" onClick={handleGoToLogin}>
                            Quay về trang đăng nhập
                        </button>
                    </div>
                )}

            </div>
        </div>
    );
};

export default Home;
