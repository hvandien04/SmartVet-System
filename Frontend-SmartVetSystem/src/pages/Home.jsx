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
    const [showGuestMessage, setShowGuestMessage] = useState(false);

    useEffect(() => {
        const groups = getSidebarGroups(user || {});
        setSidebarGroups(groups);
    }, [user]);

    useEffect(() => {
        const timer = setTimeout(() => {
            if (!user) {
                setShowGuestMessage(true);
            }
        }, 300);

        return () => clearTimeout(timer); // Xoá nếu component unmount
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
                setSearchTerm={value => setSearchTerm(value)}
            />

            <div className="main">
                <div className="page-header">
                    <h1>{activeSidebarItem}</h1>
                </div>

                {showGuestMessage && !user && (
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
