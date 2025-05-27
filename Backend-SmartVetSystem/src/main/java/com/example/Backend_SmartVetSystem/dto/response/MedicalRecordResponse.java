package com.example.Backend_SmartVetSystem.dto.response;

import com.example.Backend_SmartVetSystem.entity.Pet;
import com.example.Backend_SmartVetSystem.entity.User;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class MedicalRecordResponse {
    private PetResponse pet;
    private UserResponse user;
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
    private Boolean followUpRequired = false;
    private Instant nextVisitDate;
    private String noteForOwner;
    private String status;

}