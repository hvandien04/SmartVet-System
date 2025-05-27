package com.example.Backend_SmartVetSystem.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.ColumnDefault;

import java.time.Instant;

@Getter
@Setter
@Entity
@AllArgsConstructor
@NoArgsConstructor
@Builder
@Table(name = "appointment")
public class Appointment {
    @Id
    @Column(name = "appointment_id", nullable = false, length = 50)
    private String appointmentId;

    @Column(name = "pet_id", length = 50)
    private String petId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "owner_id")
    private com.example.Backend_SmartVetSystem.entity.Owner owner;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id")
    private com.example.Backend_SmartVetSystem.entity.User user;

    @Column(name = "appointment_time", nullable = false)
    private Instant appointmentTime;

    @Lob
    @Column(name = "reason")
    private String reason;

    @Lob
    @Column(name = "content")
    private String content;

    @ColumnDefault("'pending'")
    @Lob
    @Column(name = "status")
    private String status;

}