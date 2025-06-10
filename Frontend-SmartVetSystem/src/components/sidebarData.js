export function getSidebarGroups(user) {
    return [
        {
            title: 'Quản trị hệ thống',
            items: ['Thống kê', 'Quản lý bác sĩ', 'Quản lý thú cưng', 'Quản lý khách hàng'],
        },
        {
            title: 'Dự đoán bệnh',
            items: ['Dự đoán bệnh', 'Lịch sử dự đoán'],
        },
        {
            title: 'Hồ sơ bệnh án',
            items: ['Danh sách bệnh án', 'Xuất bệnh án'],
        },
        {
            title: 'Lịch khám bệnh',
            items: ['Lịch làm việc', 'Đặt lịch hẹn', 'Thông báo khách hàng'],
        },
        {
            title: 'User Info',
            isUserInfo: true,
            user: {
                avatarUrl: user?.avatarUrl || 'https://i.pravatar.cc/40',
                name: user?.fullName || 'Người dùng',
            },
        },
    ];
}
