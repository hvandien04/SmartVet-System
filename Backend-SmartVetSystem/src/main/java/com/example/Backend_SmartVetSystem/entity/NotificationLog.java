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
@Table(name = "notification_log")
public class NotificationLog {
    @Id
    @Column(name = "notification_id", nullable = false, length = 50)
    private String notificationId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "appointment_id")
    private Appointment appointment;

    @Column(name = "email_to", nullable = false)
    private String emailTo;

    @Column(name = "sent_time", nullable = false)
    private Instant sentTime;

    @ColumnDefault("'pending'")
    @Lob
    @Column(name = "status")
    private String status;

}