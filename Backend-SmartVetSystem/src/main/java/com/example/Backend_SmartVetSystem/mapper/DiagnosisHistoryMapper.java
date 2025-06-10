package com.example.Backend_SmartVetSystem.mapper;

import com.example.Backend_SmartVetSystem.dto.request.DiagnosisHistoryRequest;
import com.example.Backend_SmartVetSystem.dto.response.DiagnosisHistoryResponse;
import com.example.Backend_SmartVetSystem.entity.DiagnosisHistory;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.MappingTarget;

@Mapper (componentModel = "spring",uses = {UserMapper.class,MedicalRecordMapper.class})
public interface DiagnosisHistoryMapper {
    DiagnosisHistory toDiagnosisHistory(DiagnosisHistoryRequest request);

//    @Mapping(source = "pet.petId", target = "petId")
    @Mapping(source = "record", target = "medicalRecord")
    DiagnosisHistoryResponse toDiagnosisHistoryResponse(DiagnosisHistory history);
    void updateDiagnosisHistory(@MappingTarget DiagnosisHistory history, DiagnosisHistoryRequest request);
}
