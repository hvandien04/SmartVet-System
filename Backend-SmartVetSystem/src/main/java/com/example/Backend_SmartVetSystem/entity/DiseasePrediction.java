package com.example.Backend_SmartVetSystem.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.Instant;

@Getter
@Setter
@Entity
@AllArgsConstructor
@NoArgsConstructor
@Builder
@Table(name = "disease_prediction")
public class DiseasePrediction {
    @Id
    @Column(name = "prediction_id", nullable = false, length = 50)
    private String predictionId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id")
    private com.example.Backend_SmartVetSystem.entity.User user;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    @Column(name = "animal_type", length = 50)
    private String animalType;

    @Column(name = "breed", length = 50)
    private String breed;

    @Column(name = "gender", length = 10)
    private String gender;

    @Column(name = "age_years")
    private Float ageYears;

    @Column(name = "weight_kg")
    private Float weightKg;

    @Column(name = "duration_days")
    private Integer durationDays;

    @Lob
    @Column(name = "duration_category")
    private String durationCategory;

    @Lob
    @Column(name = "severity")
    private String severity;

    @Lob
    @Column(name = "season")
    private String season;

    @Lob
    @Column(name = "living_area")
    private String livingArea;

    @Column(name = "body_temperature_c")
    private Float bodyTemperatureC;

    @Column(name = "heart_rate_bpm")
    private Integer heartRateBpm;

    @Lob
    @Column(name = "description")
    private String description;

    @Column(name = "predicted_disease")
    private String predictedDisease;

    @Column(name = "probability")
    private Float probability;

}