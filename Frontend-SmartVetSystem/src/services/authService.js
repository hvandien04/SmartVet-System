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
        throw new Error(`Failed to refresh token (${res.status})`);
    }

    const text = await res.text();

    if (!text) {
        throw new Error('Empty response body');
    }

    let data;
    try {
        data = JSON.parse(text);
        // eslint-disable-next-line no-unused-vars
    } catch (err) {
        throw new Error('Invalid JSON in refresh token response');
    }

    // Lấy token từ data.result.token thay vì data.accessToken
    if (!data.result || !data.result.token) {
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

export const sendOtpToEmail = async (email) => {
    const response = await fetch('http://localhost:8080/user/forgot-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email }), // gửi { email: "..." }
    });

    if (!response.ok) {
        throw new Error('Failed to send OTP');
    }
    return true;
};

export const verifyOtpCode = async (email, code) => {
    const response = await fetch('http://localhost:8080/user/verify-code', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, code }),
    });

    if (!response.ok) {
        throw new Error('Failed to verify OTP');
    }
    const result = await response.json();
    return result.result;
};

export const resetPassword = async (email, code, newPassword) => {
    const response = await fetch('http://localhost:8080/user/reset-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, code, newPassword }),
    });

    if (!response.ok) {
        throw new Error('Failed to reset password');
    }
    const result = await response.json();
    return result.result;
};
