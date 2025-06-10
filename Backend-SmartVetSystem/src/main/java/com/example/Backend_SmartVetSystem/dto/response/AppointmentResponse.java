package com.example.Backend_SmartVetSystem.dto.response;

import com.example.Backend_SmartVetSystem.entity.Owner;
import com.example.Backend_SmartVetSystem.entity.User;
import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.ColumnDefault;

import java.time.Instant;

@AllArgsConstructor
@NoArgsConstructor
@Data
@Builder
public class AppointmentResponse {
    private String appointmentId;
    private String petId;
    private OwnerResponse owner;
    private UserResponse user;
    private Instant appointmentTime;
    private String reason;
    private String content;
    private String status;

}