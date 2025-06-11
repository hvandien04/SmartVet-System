import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import 'font-awesome/css/font-awesome.min.css';
import '@fortawesome/fontawesome-free/css/all.min.css';
import logo from '../assets/logo.png';

function getIconClass(item) {
    switch (item) {
        case 'Thống kê': return 'fa fa-chart-line';
        case 'Quản lý bác sĩ': return 'fa fa-user-md';
        case 'Quản lý thú cưng': return 'fa fa-paw';
        case 'Quản lý khách hàng': return 'fa fa-users';
        case 'Dự đoán bệnh': return 'fa fa-stethoscope';
        case 'Lịch sử dự đoán': return 'fa fa-history';
        case 'Danh sách bệnh án': return 'fa fa-file-medical';
        case 'Xuất bệnh án': return 'fa fa-file-export';
        case 'Lịch làm việc': return 'fa fa-calendar-alt';
        case 'Đặt lịch hẹn': return 'fa fa-calendar-check';
        case 'Thông báo khách hàng': return 'fa fa-bell';
        default: return 'fa fa-folder';
    }
}

function getRouteFromItem(item) {
    switch (item) {
        case 'Thống kê': return '/dashboard';
        case 'Quản lý bác sĩ': return '/doctormanagement';
        case 'Quản lý thú cưng': return '/petmanagement';
        case 'Quản lý khách hàng': return '/ownermanagement';
        case 'Dự đoán bệnh': return '/predict';
        case 'Lịch sử dự đoán': return '/predictionhistory';
        case 'Danh sách bệnh án': return '/medicalrecords';
        case 'Xuất bệnh án': return '/exportrecords';
        case 'Lịch làm việc': return '/schedule';
        case 'Đặt lịch hẹn': return '/appointment';
        case 'Thông báo khách hàng': return '/notifications';
        default: return '/';
    }
}

const Sidebar = ({ groups, activeItem, onItemClick }) => {
    const { logout } = useAuth();
    const navigate = useNavigate();
    const [showMenu, setShowMenu] = useState(false);
    const menuRef = useRef(null);

    const normalGroups = groups.filter(g => !g.isUserInfo);
    const userGroup = groups.find(g => g.isUserInfo);

    const handleClick = (item) => {
        const route = getRouteFromItem(item);
        onItemClick(item);
        navigate(route);
    };

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    // Ẩn menu khi click ra ngoài
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (menuRef.current && !menuRef.current.contains(event.target)) {
                setShowMenu(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    return (
        <div className="sidebar">
            <div className="sidebar-logo">
                <img src={logo} alt="Logo" className="logo-image" />
                <span className="logo-text">SmartVet</span>
            </div>

            {normalGroups.map(group => (
                <div key={group.title} className="sidebar-group">
                    <hr className="sidebar-separator" />
                    <h3>{group.title}</h3>
                    <ul>
                        {group.items.map(item => (
                            <li
                                key={item}
                                className={activeItem === item ? 'active' : ''}
                                onClick={() => handleClick(item)}
                            >
                                <i className={getIconClass(item)} style={{ marginRight: 8 }} />
                                {item}
                            </li>
                        ))}
                    </ul>
                </div>
            ))}

            {userGroup && userGroup.user && (
                <div className="sidebar-user-info">
                    <img src={userGroup.user.avatarUrl} alt="Avatar" className="user-avatar" />
                    <div className="user-info-text">
                        <div className="user-name">{userGroup.user.name}</div>
                        <div className="view-profile">View Profile</div>
                    </div>

                    <div className="user-settings" ref={menuRef}>
                        <button
                            className="btn-settings"
                            onClick={() => setShowMenu(!showMenu)}
                        >
                            <i className="fa fa-cog" />
                        </button>

                        {showMenu && (
                            <div className="settings-menu">
                                <button onClick={handleLogout}>
                                    <i className="fa fa-sign-out-alt" style={{ marginRight: 8 }} />
                                    Đăng xuất
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default Sidebar;
