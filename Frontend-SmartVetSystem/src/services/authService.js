// Dữ liệu giả lập người dùng
let dummyUser = {
    email: 'test@example.com',
    password: '123456',
    name: 'Test User',
    role: 'user',
    otp: '123456', // mã OTP giả lập
};

// Đăng nhập (giả lập)
export const login = async (email, password) => {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            if (email === dummyUser.email && password === dummyUser.password) {
                resolve({
                    token: 'mock-jwt-token-12345',
                    user: {
                        email: dummyUser.email,
                        name: dummyUser.name,
                        role: dummyUser.role,
                    }
                });
            } else {
                reject(new Error('Email or password is incorrect'));
            }
        }, 500);
    });
};

// Đăng ký (giả lập)
export const registerDummy = async (email, password, name) => {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            if (email === dummyUser.email) {
                reject(new Error('Email already exists'));
            } else {
                dummyUser = { email, password, name, role: 'user', otp: '123456' };
                resolve({
                    token: 'mock-jwt-token-67890',
                    user: {
                        email,
                        name,
                        role: 'user',
                    }
                });
            }
        }, 500);
    });
};

// Quên mật khẩu (giả lập)
export const forgotPassword = async (email) => {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            if (email === dummyUser.email) {
                resolve({ message: 'Password reset link sent to your email.' });
            } else {
                reject(new Error('Email not found'));
            }
        }, 500);
    });
};

// Xác nhận mã OTP (giả lập)
export const verifyEmailOtpDummy = async (email, otp) => {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            if (email === dummyUser.email) {
                if (otp === dummyUser.otp) {
                    resolve({ message: 'OTP verified successfully' });
                } else {
                    reject(new Error('Invalid OTP'));
                }
            } else {
                reject(new Error('Email not found'));
            }
        }, 500);
    });
};

// Lấy headers có chứa token (giả lập)
export const getAuthHeaders = () => {
    const token = localStorage.getItem('token');
    return {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
    };
};

// Hàm register giả lập (thay thế hàm backend)
export async function register(username, email, password) {
    // username map với name
    return registerDummy(email, password, username);
}

// Hàm verify OTP giả lập (thay thế hàm backend)
export async function verifyEmailOtp(email, otp) {
    return verifyEmailOtpDummy(email, otp);
}
