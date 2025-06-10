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
public class DiseasePredictionResponse {
    private String predictionId;
    private UserResponse user;
    private Instant createdAt;
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
    private String predictedDisease;
    private Float probability;
}
