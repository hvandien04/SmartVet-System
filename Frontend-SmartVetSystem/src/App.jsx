import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Login from './pages/Login';
import Home from './pages/Home';
import DoctorManagement from './pages/DoctorManagement';
import OwnerManagement from './pages/OwnerManagement';
import ForgotPassword from './pages/ForgotPassword';
import ResetPassword from './pages/ResetPassword';
import PetManagement from './pages/PetManagement';
import PetDetailManagement from './pages/PetDetailManagement.jsx';
import './App.css';
import { AuthProvider } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute.jsx';

function App() {
    return (
        <AuthProvider>
            <Router>
                <Routes>
                    <Route path="/" element={<Home />} />
                    <Route path="/login" element={<Login />} />
                    <Route path="/forgot-password" element={<ForgotPassword />} />
                    <Route path="/reset-password" element={<ResetPassword />} />

                    {/* Các route cần bảo vệ */}
                    <Route
                        path="/doctormanagement"
                        element={
                            <ProtectedRoute>
                                <DoctorManagement />
                            </ProtectedRoute>
                        }
                    />
                    <Route
                        path="/ownermanagement"
                        element={
                            <ProtectedRoute>
                                <OwnerManagement />
                            </ProtectedRoute>
                        }
                    />
                    <Route
                        path="/petmanagement"
                        element={
                            <ProtectedRoute>
                                <PetManagement />
                            </ProtectedRoute>
                        }
                    />
                    <Route
                        path="/petdetail/:petId"
                        element={
                            <ProtectedRoute>
                                <PetDetailManagement />
                            </ProtectedRoute>
                        }
                    />
                </Routes>
            </Router>
        </AuthProvider>
    );
}

export default App;
