package com.example.Backend_SmartVetSystem.mapper;

import com.example.Backend_SmartVetSystem.dto.request.PetCreateRequest;
import com.example.Backend_SmartVetSystem.dto.request.PetUpdateRequest;
import com.example.Backend_SmartVetSystem.dto.response.PetResponse;
import com.example.Backend_SmartVetSystem.entity.Pet;
import org.mapstruct.Mapper;
import org.mapstruct.MappingTarget;

@Mapper(componentModel = "spring",uses = {OwnerMapper.class})
public interface PetMapper {
    Pet toPet(PetCreateRequest request);
    PetResponse toPetResponse(Pet pet);
    void UpdatePet(@MappingTarget Pet pet, PetUpdateRequest request);
}
