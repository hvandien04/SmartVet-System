package com.example.Backend_SmartVetSystem.dto.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDate;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class PetUpdateRequest {
    private String name;
    private String species;
    private String breed;
    private String gender;
    private LocalDate birthDate;
    private String ownerId;
}
