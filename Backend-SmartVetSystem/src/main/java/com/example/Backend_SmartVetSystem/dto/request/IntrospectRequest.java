package com.example.Backend_SmartVetSystem.dto.request;

import lombok.*;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class IntrospectRequest {
    private String token;
    private String refreshToken;
}