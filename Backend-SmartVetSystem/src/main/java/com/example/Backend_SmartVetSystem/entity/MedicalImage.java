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
@Table(name = "medical_image")
public class MedicalImage {
    @Id
    @Column(name = "image_id", nullable = false, length = 50)
    private String imageId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "record_id")
    private com.example.Backend_SmartVetSystem.entity.MedicalRecord record;

    @Column(name = "image_url", nullable = false)
    private String imageUrl;

    @Lob
    @Column(name = "description")
    private String description;

    @Column(name = "uploaded_at", nullable = false)
    private Instant uploadedAt;

}