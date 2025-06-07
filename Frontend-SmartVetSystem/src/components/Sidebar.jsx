import React from 'react';
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

const Sidebar = ({ groups, activeItem, onItemClick }) => {
    const normalGroups = groups.filter(g => !g.isUserInfo);
    const userGroup = groups.find(g => g.isUserInfo);

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
                                onClick={() => onItemClick(item)}
                            >
                                <i className={getIconClass(item)} style={{ marginRight: 8 }} />
                                {item}
                            </li>
                        ))}
                    </ul>
                </div>
            ))}

            {userGroup && (
                <div className="sidebar-user-info">
                    <img src={userGroup.user.avatarUrl} alt="Avatar" className="user-avatar" />
                    <div className="user-info-text">
                        <div className="user-name">{userGroup.user.name}</div>
                        <div className="view-profile">View Profile</div>
                    </div>
                    <button className="btn-settings">
                        <i className="fa fa-cog" />
                    </button>
                </div>
            )}
        </div>
    );
};

export default Sidebar;
