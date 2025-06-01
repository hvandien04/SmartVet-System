import React, { useState } from 'react';
import { FaLock } from 'react-icons/fa';
import { IoEyeSharp, IoEyeOffSharp } from 'react-icons/io5';
import '../styles/Login.css';
import { FiGlobe, FiHelpCircle } from 'react-icons/fi';
import logo from '../assets/logo.png';
import imgMain from '../assets/img-main.png';
import imgSmall1 from '../assets/img-small1.png';
import imgSmall2 from '../assets/img-small2.png';

function ResetPassword() {
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);
    const [loading, setLoading] = useState(false);

    const toggleShowPassword = () => setShowPassword(!showPassword);
    const toggleShowConfirmPassword = () => setShowConfirmPassword(!showConfirmPassword);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (password !== confirmPassword) {
            alert('Passwords do not match!');
            return;
        }
        setLoading(true);
        // Giả lập delay khi reset mật khẩu thành công
        setTimeout(() => {
            setLoading(false);
            alert('Password reset successful! You can now log in.');
            window.location.href = '/login';
        }, );
    };

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
                    <h2 className="title">Reset your password 🔑</h2>
                    <p className="subtitle">Enter a new password below.</p>
                    <form className="login-form" onSubmit={handleSubmit}>
                        <div className="textbox">
                            <FaLock className="textbox-icon left" />
                            <input
                                type={showPassword ? 'text' : 'password'}
                                name="password"
                                placeholder="New password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                            />
                            {showPassword ? (
                                <IoEyeSharp
                                    className="textbox-icon right"
                                    style={{ cursor: 'pointer' }}
                                    onClick={toggleShowPassword}
                                />
                            ) : (
                                <IoEyeOffSharp
                                    className="textbox-icon right"
                                    style={{ cursor: 'pointer' }}
                                    onClick={toggleShowPassword}
                                />
                            )}
                        </div>

                        <div className="textbox">
                            <FaLock className="textbox-icon left" />
                            <input
                                type={showConfirmPassword ? 'text' : 'password'}
                                name="confirmPassword"
                                placeholder="Confirm new password"
                                value={confirmPassword}
                                onChange={(e) => setConfirmPassword(e.target.value)}
                                required
                            />
                            {showConfirmPassword ? (
                                <IoEyeSharp
                                    className="textbox-icon right"
                                    style={{ cursor: 'pointer' }}
                                    onClick={toggleShowConfirmPassword}
                                />
                            ) : (
                                <IoEyeOffSharp
                                    className="textbox-icon right"
                                    style={{ cursor: 'pointer' }}
                                    onClick={toggleShowConfirmPassword}
                                />
                            )}
                        </div>

                        <button type="submit" className="button" disabled={loading}>
                            {loading ? 'Resetting...' : 'Reset Password'}
                        </button>
                    </form>

                    <p className="footer normal">Remember your password?</p>
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

export default ResetPassword;
