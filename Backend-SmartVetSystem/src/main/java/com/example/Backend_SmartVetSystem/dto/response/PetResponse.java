package com.example.Backend_SmartVetSystem.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDate;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class PetResponse {
    private String petId;
    private String name;
    private String species;
    private String breed;
    private String gender;
    private LocalDate birthDate;
    private OwnerResponse owner;
}
