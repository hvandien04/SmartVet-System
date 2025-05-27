package com.example.Backend_SmartVetSystem.mapper;

import com.example.Backend_SmartVetSystem.dto.request.MedicalRecordRequest;
import com.example.Backend_SmartVetSystem.dto.response.MedicalRecordResponse;
import com.example.Backend_SmartVetSystem.entity.MedicalRecord;
import org.mapstruct.Mapper;
import org.mapstruct.MappingTarget;

@Mapper(componentModel = "spring", uses = {UserMapper.class, PetMapper.class})
public interface MedicalRecordMapper {
    MedicalRecord toMedicalRecord(MedicalRecordRequest medicalRecordRequest);
    MedicalRecordResponse toMedicalRecordResponse(MedicalRecord medicalRecord);
    void UpdateMedicalRecord(@MappingTarget MedicalRecord medicalRecord, MedicalRecordRequest request);
}
