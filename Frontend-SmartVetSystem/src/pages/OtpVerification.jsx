import React from 'react';
import { FaEnvelope } from 'react-icons/fa';
import { FiGlobe, FiHelpCircle } from 'react-icons/fi';
import logo from '../assets/logo.png';

function OtpVerification({
                             email,
                             otpCode,
                             setOtpCode,
                             loading,
                             onSubmit,
                             onResend,
                         }) {
    // Hàm xử lý gửi lại OTP giả lập nếu không truyền từ ngoài
    const handleResendClick = () => {
        if (typeof onResend === 'function') {
            onResend();
        } else {
            alert('OTP resent to your email (mock).');
            console.log(`Mock resend OTP to ${email}`);
        }
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
                <div className="form-area" style={{ maxWidth: 400, margin: 'auto', padding: 20 }}>
                    <h2 className="title">Confirm your email</h2>
                    <p className="subtitle">
                        Enter the confirmation code sent to: <b>{email}</b>
                    </p>
                    <form className="login-form" onSubmit={onSubmit}>
                        <div className="textbox">
                            <FaEnvelope className="textbox-icon left" />
                            <input
                                type="text"
                                name="otp"
                                placeholder="Confirmation code"
                                value={otpCode}
                                onChange={(e) => setOtpCode(e.target.value)}
                                required
                            />
                        </div>
                        <button type="submit" className="button" disabled={loading}>
                            {loading ? 'Verifying...' : 'Verify'}
                        </button>
                    </form>
                    <p className="footer normal">Didn't receive the code?</p>
                    <p
                        className="footer link"
                        onClick={handleResendClick}
                        style={{ cursor: 'pointer' }}
                    >
                        Resend code
                    </p>
                </div>
            </main>
        </div>
    );
}

export default OtpVerification;
