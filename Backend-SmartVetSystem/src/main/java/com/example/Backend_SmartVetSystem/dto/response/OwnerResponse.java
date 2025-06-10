package com.example.Backend_SmartVetSystem.dto.response;


import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class OwnerResponse {
    private String ownerId;
    private String name;
    private String email;
    private String phone;
    private String address;
}
