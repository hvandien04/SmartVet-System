import api from '../api/axiosConfig';

// Tạo thú cưng mới
export async function createPet(petData) {
    try {
        const response = await api.post('/pet', petData);
        return response.data.result; // PetResponse
        // eslint-disable-next-line no-unused-vars
    } catch (error) {
        throw new Error('Failed to create pet');
    }
}

// Lấy tất cả thú cưng
export async function fetchAllPets() {
    try {
        const response = await api.get('/pet');
        return response.data.result; // List<PetResponse>
        // eslint-disable-next-line no-unused-vars
    } catch (error) {
        throw new Error('Failed to fetch all pets');
    }
}

// Lấy thú cưng theo ID
export async function fetchPetById(petId) {
    try {
        const response = await api.get(`/pet/${petId}`);
        return response.data.result; // PetResponse
        // eslint-disable-next-line no-unused-vars
    } catch (error) {
        throw new Error('Failed to fetch pet by ID');
    }
}

// Lấy thú cưng theo ID chủ sở hữu
export async function fetchPetsByOwner(ownerId) {
    try {
        const response = await api.get(`/pet/owner/${ownerId}`);

        return response.data.result; // List<PetResponse>
        // eslint-disable-next-line no-unused-vars
    } catch (error) {
        throw new Error('Failed to fetch pets by owner');
    }
}

// Cập nhật thú cưng
export async function updatePet(petId, petData) {
    try {
        const response = await api.put(`/pet/${petId}`, petData);
        return response.data.result; // PetResponse
        // eslint-disable-next-line no-unused-vars
    } catch (error) {
        throw new Error('Failed to update pet');
    }
}

// Xóa thú cưng
export async function deletePet(petId) {
    try {
        const response = await api.delete(`/pet/${petId}`);
        return response.data.result; // String message
        // eslint-disable-next-line no-unused-vars
    } catch (error) {
        throw new Error('Failed to delete pet');
    }
}
