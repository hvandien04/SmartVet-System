package com.example.Backend_SmartVetSystem.controller;

import com.example.Backend_SmartVetSystem.dto.request.ApiResponse;
import com.example.Backend_SmartVetSystem.dto.request.MedicalRecordRequest;
import com.example.Backend_SmartVetSystem.dto.response.MedicalRecordResponse;
import com.example.Backend_SmartVetSystem.service.MedicalRecordService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RequiredArgsConstructor
@RestController
@RequestMapping("/medical-record")
public class MedicalRecordController {
    private final MedicalRecordService medicalRecordService;

    @PostMapping
    ApiResponse<MedicalRecordResponse> createMedicalRecord(@RequestBody MedicalRecordRequest medicalRecordRequest) {
        return ApiResponse.<MedicalRecordResponse>builder()
                .result(medicalRecordService.createMedicalRecord(medicalRecordRequest))
                .build();
    }

    @GetMapping
    ApiResponse<List<MedicalRecordResponse>> getAllMedicalRecords() {
        return ApiResponse.<List<MedicalRecordResponse>>builder()
                .result(medicalRecordService.getAllMedicalRecords())
                .build();
    }

    @GetMapping("/{recordId}")
    ApiResponse<MedicalRecordResponse> getMedicalRecord(@PathVariable String recordId) {
        return ApiResponse.<MedicalRecordResponse>builder()
                .result(medicalRecordService.getMedicalRecord(recordId))
                .build();
    }

    @PutMapping("/{recordId}")
    ApiResponse<MedicalRecordResponse> updateMedicalRecord(@PathVariable String recordId, @RequestBody MedicalRecordRequest medicalRecordRequest) {
        return ApiResponse.<MedicalRecordResponse>builder()
                .result(medicalRecordService.updateMedicalRecord(recordId, medicalRecordRequest))
                .build();
    }

    @DeleteMapping("/{recordId}")
    ApiResponse<String> deleteMedicalRecord(@PathVariable String recordId) {
        return ApiResponse.<String>builder()
                .result(medicalRecordService.deleteMedicalRecord(recordId))
                .build();
    }

}
