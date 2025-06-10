package com.example.Backend_SmartVetSystem.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class DiagnosisHistoryResponse {
    private String diagnosisId;
    private MedicalRecordResponse medicalRecord;
    private Instant createdAt;
}
