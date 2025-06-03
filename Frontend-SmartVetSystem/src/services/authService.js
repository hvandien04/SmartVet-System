// Dữ liệu người dùng giả lập (chỉ dùng demo)
let dummyUser = {
    email: 'test@example.com',
    password: '123456',
    name: 'Test User',
    role: 'user',
    otp: '123456', // mã OTP giả lập
};

export async function fetchUserInfo(accessToken) {
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

// Hàm login, trả về access token + user info
export const login = async (username, password) => {
    const response = await fetch('http://localhost:8080/auth', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include', // gửi cookie (refresh token)
        body: JSON.stringify({ username, password }),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Login failed');
    }

    const data = await response.json();
    const token = data.result.token;

    // Gọi fetchUserInfo để lấy thông tin người dùng từ /user endpoint
    const user = await fetchUserInfo(token);

    // Log ra để kiểm tra
    console.log('User info from fetchUserInfo:', user);
    console.log('Access token:', token);

    return {
        token,
        user
    };
};

export const refreshToken = async () => {
    const res = await fetch('http://localhost:8080/auth/refresh-token', {
        method: 'POST',
        credentials: 'include',
    });

    if (!res.ok) {
        console.error('Refresh token failed with status:', res.status);
        throw new Error(`Failed to refresh token (${res.status})`);
    }

    const text = await res.text();
    console.log('Raw refresh token response text:', text);

    if (!text) {
        console.error('Refresh token response body empty');
        throw new Error('Empty response body');
    }

    let data;
    try {
        data = JSON.parse(text);
    } catch (err) {
        console.error('Failed to parse JSON:', err);
        throw new Error('Invalid JSON in refresh token response');
    }

    console.log('Parsed refresh token response:', data);

    // Lấy token từ data.result.token thay vì data.accessToken
    if (!data.result || !data.result.token) {
        console.error('token missing in refresh token response');
        throw new Error('No token in refresh token response');
    }

    return data.result.token;
};
function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(';').shift();
}


// Hàm logout nhận accessToken từ context
export async function logout(accessToken) {
    const refreshToken = getCookie('refresh_token');

    const introspectRequest = {
        token: accessToken || '',
        refreshToken: refreshToken || '',
    };

    const response = await fetch('http://localhost:8080/auth/logout', {
        method: 'POST',
        credentials: 'include',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${accessToken}`,
        },
        body: JSON.stringify(introspectRequest),
    });

    if (!response.ok) {
        const errorText = await response.text();
        console.error('Logout failed response:', errorText);
        throw new Error('Logout failed');
    }

    return true;
}


// Các hàm giả lập cho demo, bạn có thể dùng hoặc bỏ qua
const registerDummy = async (email, password, name) => {
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

const verifyEmailOtpDummy = async (email, otp) => {
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

export const getAuthHeaders = (token) => ({
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
});

export async function register(username, email, password) {
    return registerDummy(email, password, username);
}

export async function verifyEmailOtp(email, otp) {
    return verifyEmailOtpDummy(email, otp);
}
