import React, { useState } from 'react';
import '../styles/Login.css';
import { FaEnvelope } from 'react-icons/fa';
import { FiGlobe, FiHelpCircle } from 'react-icons/fi';
import logo from '../assets/logo.png';
import imgMain from '../assets/img-main.png';
import imgSmall1 from '../assets/img-small1.png';
import imgSmall2 from '../assets/img-small2.png';
import OtpVerification from './OtpVerification'; // Import component OTP

import { sendOtpToEmail, verifyOtpCode } from '../services/userService';

function ForgotPassword() {
    const [email, setEmail] = useState('');
    const [otpSent, setOtpSent] = useState(false);
    const [otpCode, setOtpCode] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSendOtp = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            await sendOtpToEmail(email);
            setOtpSent(true);
            alert(`Request successful! A confirmation code has been sent to your email: ${email}\n`);
        } catch (error) {
            alert(error.message || 'Failed to send OTP');
            console.error(error);
        }
        setLoading(false);
    };

    const handleVerifyOtp = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            const isValid = await verifyOtpCode(email, otpCode);
            if (isValid) {
                alert('OTP verified! Redirecting to reset password page...');
                window.location.href = `/reset-password?email=${email}&code=${otpCode}`;
            } else {
                alert('Incorrect OTP. Please try again.');
            }
        } catch (error) {
            alert('Verification failed');
            console.error(error);
        }
        setLoading(false);
    };

    const handleResend = async () => {
        setLoading(true);
        try {
            await sendOtpToEmail(email);
            alert('OTP resent to your email.');
        } catch (error) {
            alert(error.message || 'Failed to resend OTP');
        }
        setLoading(false);
    };

    // Nếu đã gửi OTP → hiển thị form xác nhận OTP
    if (otpSent) {
        return (
            <OtpVerification
                email={email}
                otpCode={otpCode}
                setOtpCode={setOtpCode}
                loading={loading}
                onSubmit={handleVerifyOtp}
                onResend={handleResend}
            />
        );
    }

    // Giao diện gửi email (mặc định)
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
                    <h2 className="title">Forgot your password? 🔒</h2>
                    <p className="subtitle">We'll send you a code to reset it.</p>
                    <form className="login-form" onSubmit={handleSendOtp}>
                        <div className="textbox">
                            <FaEnvelope className="textbox-icon left" />
                            <input
                                type="email"
                                name="email"
                                placeholder="Enter your email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                            />
                        </div>
                        <button type="submit" className="button" disabled={loading}>
                            {loading ? 'Sending...' : 'Send code'}
                        </button>
                    </form>
                    <p className="footer normal">Remembered your password?</p>
                    <p
                        className="footer link"
                        onClick={() => (window.location.href = '/login')}
                        style={{ cursor: 'pointer' }}
                    >
                        Log in
                    </p>
                </div>
            </main>
        </div>
    );
}

export default ForgotPassword;
