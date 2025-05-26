package com.example.Backend_SmartVetSystem.dto.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@NoArgsConstructor
@Data
@AllArgsConstructor
@Builder
public class UserUpdateRequest {
    private String email;
    private String password;
    private String role;
    private String fullName;
    private String phone;
    private String address;
}
