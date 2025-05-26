package com.example.Backend_SmartVetSystem.dto.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDate;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class PetCreateRequest {
    private String name;
    private String species;
    private String breed;
    private String gender;
    private LocalDate birthDate;
    private OwnerRequest owner;
    private String ownerId;
}
