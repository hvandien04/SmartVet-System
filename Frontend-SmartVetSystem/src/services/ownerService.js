import api from '../api/axiosConfig';

// Lấy toàn bộ danh sách chủ sở hữu
export async function fetchAllOwners() {
    try {
        const response = await api.get('/owner');
        return response.data.result; // List<OwnerResponse>
        // eslint-disable-next-line no-unused-vars
    } catch (error) {
        throw new Error('Failed to fetch all owners');
    }
}

// Lấy thông tin của một owner cụ thể theo ID
export async function fetchOwnerById(ownerId) {
    try {
        const response = await api.get(`/owner/${ownerId}`);
        return response.data.result; // OwnerResponse
        // eslint-disable-next-line no-unused-vars
    } catch (error) {
        throw new Error(`Failed to fetch owner with ID ${ownerId}`);
    }
}

// Tạo mới một owner
export async function createOwner(ownerRequest) {
    try {
        const response = await api.post('/owner', ownerRequest);
        return response.data.result; // OwnerResponse
        // eslint-disable-next-line no-unused-vars
    } catch (error) {
        throw new Error('Failed to create owner');
    }
}

// Cập nhật thông tin một owner theo ID
export async function updateOwner(ownerId, ownerRequest) {
    try {
        const response = await api.put(`/owner/${ownerId}`, ownerRequest);
        return response.data.result; // OwnerResponse
        // eslint-disable-next-line no-unused-vars
    } catch (error) {
        throw new Error(`Failed to update owner with ID ${ownerId}`);
    }
}

// Xóa một owner theo ID
export async function deleteOwner(ownerId) {
    try {
        const response = await api.delete(`/owner/${ownerId}`);
        return response.data.result; // String (message)
        // eslint-disable-next-line no-unused-vars
    } catch (error) {
        throw new Error(`Failed to delete owner with ID ${ownerId}`);
    }
}
