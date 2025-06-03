package com.example.Backend_SmartVetSystem.dto.request;

import com.example.Backend_SmartVetSystem.entity.Owner;
import com.example.Backend_SmartVetSystem.entity.User;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;

@AllArgsConstructor
@NoArgsConstructor
@Data
@Builder
public class AppointmentRequest {
    private String petId;
    private String ownerId;
    private String userId;
    private Instant appointmentTime;
    private String reason;
    private String content;
    private String status;

}