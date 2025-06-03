package com.example.Backend_SmartVetSystem.mapper;

import com.example.Backend_SmartVetSystem.dto.request.MedicalImageRequest;
import com.example.Backend_SmartVetSystem.dto.response.MedicalImageResponse;
import com.example.Backend_SmartVetSystem.dto.response.MedicalRecordResponse;
import com.example.Backend_SmartVetSystem.entity.MedicalImage;
import org.mapstruct.Mapper;

@Mapper(componentModel = "spring")
public interface MedicalImageMapper {
    MedicalImage toMedicalImage(MedicalImageRequest medicalRecordResponse);
    MedicalImageResponse toMedicalImageResponse(MedicalImage image);
}
