package com.example.Backend_SmartVetSystem.dto.request;

import lombok.*;

import java.time.Instant;
import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class MedicalRecordRequest {
    private String petId;
    private String userId;
    private Instant visitDate;
    private String symptoms;
    private String diagnosisSummary;
    private String clinicalTestResults;
    private String animalType;
    private String breed;
    private String gender;
    private Float ageYears;
    private Float weightKg;
    private Integer durationDays;
    private String durationCategory;
    private String severity;
    private String season;
    private String livingArea;
    private Float bodyTemperatureC;
    private Integer heartRateBpm;
    private String description;
    private String treatmentPlan;
    private String medicationsPrescribed;
    private Boolean followUpRequired;
    private Instant nextVisitDate;
    private String noteForOwner;
    private String status;
    private List<MedicalImageRequest> medicalImageRequest;
}