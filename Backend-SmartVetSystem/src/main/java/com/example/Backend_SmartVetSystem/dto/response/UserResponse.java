package com.example.Backend_SmartVetSystem.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@NoArgsConstructor
@Data
@AllArgsConstructor
@Builder
public class UserResponse {
    private String userId;
    private String username;
    private String email;
    private String role;
    private String fullName;
    private String phone;
    private String address;
}
