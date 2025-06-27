import api from '../api/axiosConfig';

// GET: Lấy tất cả cuộc hẹn
export async function fetchAllAppointments() {
  try {
    const response = await api.get('/appointment');
    console.log('fetchAllAppointments response:', response.data);
    return response.data.result;
  } catch (error) {
    console.error('fetchAllAppointments error:', {
      status: error.response?.status,
      data: error.response?.data,
      message: error.message,
    });
    throw new Error('Không thể tải danh sách cuộc hẹn');
  }
}

// GET: Lấy chi tiết cuộc hẹn
export async function fetchAppointmentById(appointmentId) {
  try {
    const response = await api.get(`/appointment/${appointmentId}`);
    console.log('fetchAppointmentById response:', response.data);
    return response.data.result;
  } catch (error) {
    console.error('fetchAppointmentById error:', {
      status: error.response?.status,
      data: error.response?.data,
      message: error.message,
    });
    throw new Error(`Không thể tải chi tiết cuộc hẹn: ${appointmentId}`);
  }
}

// POST: Tạo mới cuộc hẹn
export async function createAppointment(data) {
  try {
    console.log('createAppointment data:', data);
    const response = await api.post('/appointment', data);
    console.log('createAppointment response:', response.data);
    return response.data.result;
  } catch (error) {
    console.error('createAppointment error:', {
      status: error.response?.status,
      data: error.response?.data,
      message: error.message,
    });
    throw new Error('Không thể tạo cuộc hẹn');
  }
}

// PUT: Cập nhật cuộc hẹn
export async function updateAppointment(appointmentId, data) {
  try {
    console.log('updateAppointment data:', data);
    const response = await api.put(`/appointment/${appointmentId}`, data);
    console.log('updateAppointment response:', response.data);
    return response.data.result;
  } catch (error) {
    const errorMessage = error.response?.data?.message || `Không thể cập nhật cuộc hẹn: ${appointmentId}`;
    console.error('updateAppointment error:', {
      status: error.response?.status,
      data: error.response?.data,
      message: errorMessage,
    });
    throw new Error(errorMessage);
  }
}

// DELETE: Xóa cuộc hẹn
export async function deleteAppointment(appointmentId) {
  try {
    const response = await api.delete(`/appointment/${appointmentId}`);
    console.log('deleteAppointment response:', response.data);
    return response.data.result;
  } catch (error) {
    console.error('deleteAppointment error:', {
      status: error.response?.status,
      data: error.response?.data,
      message: error.message,
    });
    throw new Error(`Không thể xóa cuộc hẹn: ${appointmentId}`);
  }
}
// GET: Lấy danh sách cuộc hẹn theo khoảng thời gian
export async function fetchAppointmentsByTime(from, to) {
  try {
    const response = await api.get('/appointment/by-time', {
      params: { from, to },
    });
    console.log('fetchAppointmentsByTime response:', response.data);
    return response.data.result; // nếu API trả về dạng { result: [...] }
  } catch (error) {
    console.error('fetchAppointmentsByTime error:', {
      status: error.response?.status,
      data: error.response?.data,
      message: error.message,
    });
    throw new Error('Không thể tải cuộc hẹn theo thời gian');
  }
}
