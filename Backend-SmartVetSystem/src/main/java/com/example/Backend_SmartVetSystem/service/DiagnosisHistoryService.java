package com.example.Backend_SmartVetSystem.service;

import com.example.Backend_SmartVetSystem.dto.request.DiagnosisHistoryRequest;
import com.example.Backend_SmartVetSystem.dto.response.DiagnosisHistoryResponse;
import com.example.Backend_SmartVetSystem.entity.DiagnosisHistory;
import com.example.Backend_SmartVetSystem.entity.MedicalRecord;
import com.example.Backend_SmartVetSystem.entity.Pet;
import com.example.Backend_SmartVetSystem.entity.User;
import com.example.Backend_SmartVetSystem.exception.AppException;
import com.example.Backend_SmartVetSystem.exception.ErrorCode;
import com.example.Backend_SmartVetSystem.mapper.DiagnosisHistoryMapper;
import com.example.Backend_SmartVetSystem.repository.DiagnosisHistoryRepository;
import com.example.Backend_SmartVetSystem.repository.MedicalRecordRepository;
import com.example.Backend_SmartVetSystem.repository.PetRepository;
import com.example.Backend_SmartVetSystem.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class DiagnosisHistoryService {
    private final DiagnosisHistoryRepository diagnosisHistoryRepository;
    private final IdGeneratorService idGeneratorService;
    private final DiagnosisHistoryMapper diagnosisHistoryMapper;
    private final PetRepository petRepository;
    private final UserRepository userRepository;
    private final MedicalRecordRepository medicalRecordRepository;

    public DiagnosisHistoryResponse CreateDiagnosisHistory(DiagnosisHistoryRequest request) {
        DiagnosisHistory diagnosisHistory = diagnosisHistoryMapper.toDiagnosisHistory(request);
        diagnosisHistory.setDiagnosisId(idGeneratorService.generateRandomId("DH",diagnosisHistoryRepository::existsById));
        Pet pet = petRepository.findById(request.getPetId()).orElseThrow(()-> new AppException(ErrorCode.PET_NOT_FOUND));
        diagnosisHistory.setPet(pet);
        User user = userRepository.findById(request.getUserId()).orElseThrow(()-> new AppException(ErrorCode.USER_NOT_FOUND));
        diagnosisHistory.setUser(user);
        MedicalRecord medicalRecord = medicalRecordRepository.findById(request.getRecordId()).orElseThrow(()-> new AppException(ErrorCode.MEDICAL_RECORD_NOT_FOUND));
        diagnosisHistory.setRecord(medicalRecord);
        diagnosisHistory.setCreatedAt(Instant.now().plus(7, ChronoUnit.HOURS));
        return diagnosisHistoryMapper.toDiagnosisHistoryResponse(diagnosisHistoryRepository.save(diagnosisHistory));
    }

    public DiagnosisHistoryResponse updateDiagnosisHistory(String diagnosisId,DiagnosisHistoryRequest request) {
        DiagnosisHistory diagnosisHistory = diagnosisHistoryRepository.findById(diagnosisId).orElseThrow(()-> new AppException(ErrorCode.DIAGNOSTIC_HISTORY_NOT_FOUND));
        diagnosisHistoryMapper.updateDiagnosisHistory(diagnosisHistory,request);
        return diagnosisHistoryMapper.toDiagnosisHistoryResponse(diagnosisHistoryRepository.save(diagnosisHistory));
    }



    public List<DiagnosisHistoryResponse> getAllDiagnosisHistory() {
        return diagnosisHistoryRepository.findAll().stream().map(diagnosisHistoryMapper::toDiagnosisHistoryResponse).collect(Collectors.toList());
    }

    public List<DiagnosisHistoryResponse> findTop10ByOrderByCreatedAtDesc() {
        return diagnosisHistoryRepository.findTop10ByOrderByCreatedAtDesc().stream().map(diagnosisHistoryMapper::toDiagnosisHistoryResponse).collect(Collectors.toList());
    }
}
