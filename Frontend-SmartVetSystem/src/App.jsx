import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Login from './pages/Login';
import Home from './pages/Home';
import Register from './pages/Register';
import ForgotPassword from './pages/ForgotPassword'; // 👈 Thêm dòng này
import ResetPassword from './pages/ResetPassword'; // 👈 Thêm dòng này
import './App.css';
import { AuthProvider } from './context/AuthContext'; // 👈 Thêm dòng này

function App() {
    return (
        <AuthProvider>
            <Router>
                <Routes>
                    <Route path="/" element={<Home />} />
                    <Route path="/login" element={<Login />} />
                    <Route path="/register" element={<Register />} />
                    <Route path="/forgot-password" element={<ForgotPassword />} />
                    <Route path="/reset-password" element={<ResetPassword />} />
                </Routes>
            </Router>
        </AuthProvider> // ✅ Đóng thẻ đúng cách
    );
}

export default App;
