package com.example.Backend_SmartVetSystem.service;

import com.example.Backend_SmartVetSystem.dto.request.MedicalRecordRequest;
import com.example.Backend_SmartVetSystem.dto.response.MedicalRecordResponse;
import com.example.Backend_SmartVetSystem.entity.MedicalImage;
import com.example.Backend_SmartVetSystem.entity.MedicalRecord;
import com.example.Backend_SmartVetSystem.entity.Pet;
import com.example.Backend_SmartVetSystem.entity.User;
import com.example.Backend_SmartVetSystem.exception.AppException;
import com.example.Backend_SmartVetSystem.exception.ErrorCode;
import com.example.Backend_SmartVetSystem.mapper.MedicalImageMapper;
import com.example.Backend_SmartVetSystem.mapper.MedicalRecordMapper;
import com.example.Backend_SmartVetSystem.repository.*;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class MedicalRecordService {
    private final MedicalRecordRepository medicalRecordRepository;
    private final IdGeneratorService idGeneratorService;
    private final MedicalRecordMapper medicalRecordMapper;
    private final UserRepository userRepository;
    private final PetRepository petRepository;
    private final MedicalImageMapper medicalImageMapper;
    private final MedicalImageRepository medicalImageRepository;
    private final DiagnosisHistoryRepository diagnosisHistoryRepository;

    public MedicalRecordResponse createMedicalRecord(MedicalRecordRequest medicalRecordRequest) {
        MedicalRecord medicalRecord = medicalRecordMapper.toMedicalRecord(medicalRecordRequest);
        medicalRecord.setRecordId(idGeneratorService.generateRandomId("R",medicalRecordRepository::existsById));
        User user = userRepository.findById(medicalRecordRequest.getUserId()).orElseThrow(()-> new AppException(ErrorCode.USER_NOT_FOUND));
        Pet pet = petRepository.findById(medicalRecordRequest.getPetId()).orElseThrow(()-> new AppException(ErrorCode.PET_NOT_FOUND));
        medicalRecord.setPet(pet);
        medicalRecord.setUser(user);
        medicalRecord = medicalRecordRepository.save(medicalRecord);

        if(medicalRecordRequest.getMedicalImageRequest() != null) {
            MedicalRecord finalMedicalRecord = medicalRecord;
            List<MedicalImage> images = medicalRecordRequest.getMedicalImageRequest().stream()
                    .map(img -> {
                        MedicalImage medicalImage = medicalImageMapper.toMedicalImage(img);
                        medicalImage.setImageId(idGeneratorService.generateRandomId("IMG",medicalRecordRepository::existsById));
                        medicalImage.setUploadedAt(Instant.now().plus(7, ChronoUnit.HOURS));
                        medicalImage.setRecord(finalMedicalRecord);
                        return medicalImage;
                    })
                    .toList();
            medicalImageRepository.saveAll(images);
        }
        return medicalRecordMapper.toMedicalRecordResponse(medicalRecord);
    }

    public MedicalRecordResponse updateMedicalRecord(String recordId, MedicalRecordRequest medicalRecordRequest) {
        MedicalRecord record = medicalRecordRepository.findById(recordId).orElseThrow(()-> new AppException(ErrorCode.MEDICAL_RECORD_NOT_FOUND));
        medicalRecordMapper.UpdateMedicalRecord(record,medicalRecordRequest);
        if(medicalRecordRequest.getPetId()!=null){
            Pet pet = petRepository.findById(medicalRecordRequest.getPetId()).orElseThrow(()-> new AppException(ErrorCode.PET_NOT_FOUND));
            record.setPet(pet);
        }
        if(medicalRecordRequest.getUserId()!=null){
            User user = userRepository.findById(medicalRecordRequest.getUserId()).orElseThrow(()-> new AppException(ErrorCode.USER_NOT_FOUND));
            record.setUser(user);
        }
        return medicalRecordMapper.toMedicalRecordResponse(medicalRecordRepository.save(record));
    }

    public List<MedicalRecordResponse> getAllMedicalRecords() {
        return medicalRecordRepository.findAll().stream().map(medicalRecordMapper::toMedicalRecordResponse).collect(Collectors.toList());
    }

    public MedicalRecordResponse getMedicalRecord(String recordId) {
        return medicalRecordMapper.toMedicalRecordResponse(medicalRecordRepository.findById(recordId).orElseThrow(()-> new AppException(ErrorCode.MEDICAL_RECORD_NOT_FOUND)));
    }

    public List<MedicalRecordResponse> getMedicalRecordByVisitDate(LocalDate visitDate) {
        return medicalRecordRepository.findByVisitDate(visitDate).stream().map(medicalRecordMapper::toMedicalRecordResponse).collect(Collectors.toList());
    }

    public List<MedicalRecordResponse> getMedicalRecordByPet(String petId) {
        Pet pet = petRepository.findById(petId).orElseThrow(()-> new AppException(ErrorCode.PET_NOT_FOUND));
        return medicalRecordRepository.findByPet(pet).stream().map(medicalRecordMapper::toMedicalRecordResponse).collect(Collectors.toList());
    }

    @Transactional
    @PreAuthorize("hasRole('ADMIN')")
    public String deleteMedicalRecord(String recordId) {
        MedicalRecord medicalRecord = medicalRecordRepository.findById(recordId).orElseThrow(()-> new AppException(ErrorCode.MEDICAL_RECORD_NOT_FOUND));
        diagnosisHistoryRepository.deleteAllByRecord(medicalRecord);
        medicalImageRepository.deleteAllByRecord(medicalRecord);
        medicalRecordRepository.deleteById(recordId);
        return "Medical Record has Id " + recordId + " deleted";
    }
}
