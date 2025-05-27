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
@Table(name = "diagnosis_history")
public class DiagnosisHistory {
    @Id
    @Column(name = "diagnosis_id", nullable = false, length = 50)
    private String diagnosisId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id")
    private com.example.Backend_SmartVetSystem.entity.User user;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "pet_id")
    private com.example.Backend_SmartVetSystem.entity.Pet pet;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "record_id")
    private com.example.Backend_SmartVetSystem.entity.MedicalRecord record;

}