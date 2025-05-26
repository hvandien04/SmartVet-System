package com.example.Backend_SmartVetSystem.mapper;

import com.example.Backend_SmartVetSystem.dto.request.OwnerRequest;
import com.example.Backend_SmartVetSystem.dto.response.OwnerResponse;
import com.example.Backend_SmartVetSystem.entity.Owner;
import org.mapstruct.Mapper;
import org.mapstruct.MappingTarget;

@Mapper(componentModel = "spring")
public interface OwnerMapper {
    Owner toOwner(OwnerRequest request);
    OwnerResponse toOwnerResponse(Owner owner);


    void UpdateOwner(@MappingTarget Owner owner, OwnerRequest response);
}
