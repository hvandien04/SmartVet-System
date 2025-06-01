import React, { useState } from 'react';
import '../styles/Login.css';
import { FaEnvelope, FaLock, FaUser } from 'react-icons/fa';
import { IoEyeSharp, IoEyeOffSharp } from 'react-icons/io5';
import { FiGlobe, FiHelpCircle } from 'react-icons/fi';
import logo from '../assets/logo.png';
import imgMain from '../assets/img-main.png';
import imgSmall1 from '../assets/img-small1.png';
import imgSmall2 from '../assets/img-small2.png';
import OtpVerification from './OtpVerification.jsx'; // import component OTP

function Register() {
    const [formData, setFormData] = useState({
        username: '',
        email: '',
        password: '',
        confirmPassword: '',
    });
    const [showPassword, setShowPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);

    const [isRegistered, setIsRegistered] = useState(false);
    const [otpCode, setOtpCode] = useState('');
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState(''); // Thông báo lỗi OTP

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const toggleShowPassword = () => {
        setShowPassword(!showPassword);
    };

    const toggleShowConfirmPassword = () => {
        setShowConfirmPassword(!showConfirmPassword);
    };

    // Giả lập đăng ký thành công (không gọi API)
    const handleRegisterSubmit = (e) => {
        e.preventDefault();

        if (formData.password !== formData.confirmPassword) {
            alert('Passwords do not match!');
            return;
        }

        setLoading(true);
        setTimeout(() => {
            setLoading(false);
            setIsRegistered(true);
            alert(`Register successful! A confirmation code has been sent to your email: ${formData.email}\nUse code 123456 to test.`);
        }, );
    };

    // Giả lập verify OTP (mã đúng là '123456')
    const handleOtpSubmit = (e) => {
        e.preventDefault();

        if (!otpCode) {
            alert('Please enter the confirmation code');
            return;
        }

        setLoading(true);
        setMessage('');
        setTimeout(() => {
            setLoading(false);
            if (otpCode === '123456') {
                alert('Email verified successfully! You can now log in.');
                window.location.href = '/login';
                // Reset trạng thái để test lại
                setIsRegistered(false);
                setOtpCode('');
                setFormData({ username: '', email: '', password: '', confirmPassword: '' });
            } else {
                setMessage('Invalid confirmation code');
            }
        }, );
    };

    // Giả lập resend code
    const handleResendCode = () => {
        setLoading(true);
        setMessage('');
        setTimeout(() => {
            setLoading(false);
            alert('Confirmation code resent! Use code 123456');
        }, );
    };

    if (isRegistered) {
        return (
            <OtpVerification
                email={formData.email}
                otpCode={otpCode}
                setOtpCode={setOtpCode}
                loading={loading}
                onSubmit={handleOtpSubmit}
                onResend={handleResendCode}
                message={message} // Nếu bạn muốn hiện thông báo lỗi OTP trong component OtpVerification
            />
        );
    }

    // Màn hình đăng ký ban đầu (giữ nguyên UI)
    return (
        <div className="login-wrapper">
            <header className="header">
                <div className="left-section">
                    <img src={logo} alt="Logo" className="logo" />
                    <p className="heading">SmartVert</p>
                </div>
                <div className="icons">
                    <FiGlobe className="icon" />
                    <FiHelpCircle className="icon" />
                </div>
            </header>

            <main className="main-content">
                <div className="visual-area">
                    <img src={imgMain} alt="Main" className="hero" />
                    <img src={imgSmall1} alt="Small1" className="icon1" />
                    <img src={imgSmall2} alt="Small2" className="icon2" />
                    <div className="rectangle large"></div>
                    <div className="rectangle small"></div>
                    <div className="oval green"></div>
                    <div className="oval pink"></div>
                </div>

                <div className="form-area">
                    <h2 className="title">Create an account 👋</h2>
                    <p className="subtitle">Join SmartVert today</p>
                    <form className="login-form" onSubmit={handleRegisterSubmit}>
                        <div className="textbox">
                            <FaUser className="textbox-icon left" />
                            <input
                                type="text"
                                name="username"
                                placeholder="Your username"
                                value={formData.username}
                                onChange={handleChange}
                                required
                            />
                        </div>
                        <div className="textbox">
                            <FaEnvelope className="textbox-icon left" />
                            <input
                                type="email"
                                name="email"
                                placeholder="Your email"
                                value={formData.email}
                                onChange={handleChange}
                                required
                            />
                        </div>
                        <div className="textbox">
                            <FaLock className="textbox-icon left" />
                            {showPassword ? (
                                <IoEyeSharp className="textbox-icon right" onClick={toggleShowPassword} />
                            ) : (
                                <IoEyeOffSharp className="textbox-icon right" onClick={toggleShowPassword} />
                            )}
                            <input
                                type={showPassword ? 'text' : 'password'}
                                name="password"
                                placeholder="Create a password"
                                value={formData.password}
                                onChange={handleChange}
                                required
                            />
                        </div>
                        <div className="textbox">
                            <FaLock className="textbox-icon left" />
                            {showConfirmPassword ? (
                                <IoEyeSharp className="textbox-icon right" onClick={toggleShowConfirmPassword} />
                            ) : (
                                <IoEyeOffSharp className="textbox-icon right" onClick={toggleShowConfirmPassword} />
                            )}
                            <input
                                type={showConfirmPassword ? 'text' : 'password'}
                                name="confirmPassword"
                                placeholder="Confirm your password"
                                value={formData.confirmPassword}
                                onChange={handleChange}
                                required
                            />
                        </div>
                        <button type="submit" className="button" disabled={loading}>
                            {loading ? 'Registering...' : 'Register'}
                        </button>
                    </form>
                    <p className="footer normal">Already have an account?</p>
                    <p className="footer link" onClick={() => (window.location.href = '/login')}>
                        Log in
                    </p>
                </div>
            </main>
        </div>
    );
}

export default Register;
