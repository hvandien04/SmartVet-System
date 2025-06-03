package com.example.Backend_SmartVetSystem.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.ColumnDefault;

import java.time.Instant;
import java.util.List;

@Getter
@Setter
@Entity
@AllArgsConstructor
@NoArgsConstructor
@Builder
@Table(name = "medical_record")
public class MedicalRecord {
    @Id
    @Column(name = "record_id", nullable = false, length = 50)
    private String recordId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "pet_id")
    private com.example.Backend_SmartVetSystem.entity.Pet pet;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id")
    private com.example.Backend_SmartVetSystem.entity.User user;

    @Column(name = "visit_date", nullable = false)
    private Instant visitDate;

    @Lob
    @Column(name = "symptoms")
    private String symptoms;

    @Lob
    @Column(name = "diagnosis_summary")
    private String diagnosisSummary;

    @Lob
    @Column(name = "clinical_test_results")
    private String clinicalTestResults;

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

    @Lob
    @Column(name = "treatment_plan")
    private String treatmentPlan;

    @Lob
    @Column(name = "medications_prescribed")
    private String medicationsPrescribed;

    @ColumnDefault("0")
    @Column(name = "follow_up_required")
    private Boolean followUpRequired;

    @Column(name = "next_visit_date")
    private Instant nextVisitDate;

    @Lob
    @Column(name = "note_for_owner")
    private String noteForOwner;

    @ColumnDefault("'ongoing'")
    @Lob
    @Column(name = "status")
    private String status;

    @OneToMany(mappedBy = "record", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private List<MedicalImage> medicalImages;

}