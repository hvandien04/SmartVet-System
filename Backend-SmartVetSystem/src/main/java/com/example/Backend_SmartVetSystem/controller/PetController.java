package com.example.Backend_SmartVetSystem.controller;

import com.example.Backend_SmartVetSystem.dto.request.ApiResponse;
import com.example.Backend_SmartVetSystem.dto.request.PetCreateRequest;
import com.example.Backend_SmartVetSystem.dto.request.PetUpdateRequest;
import com.example.Backend_SmartVetSystem.dto.response.PetResponse;
import com.example.Backend_SmartVetSystem.service.PetService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/pet")
public class PetController {
    private final PetService petService;

    @PostMapping
    ApiResponse<PetResponse> CreatePet(@RequestBody PetCreateRequest request) {
        return ApiResponse.<PetResponse>builder()
                .result(petService.createPet(request))
                .build();
    }
    @GetMapping
    ApiResponse<List<PetResponse>> GetAllPets() {
        return ApiResponse.<List<PetResponse>>builder()
                .result(petService.GetAllPets())
                .build();
    }

    @GetMapping("/{PetId}")
    ApiResponse<PetResponse> GetPet(@PathVariable String PetId) {
        return ApiResponse.<PetResponse>builder()
                .result(petService.getPet(PetId))
                .build();
    }

    @GetMapping("/owner/{ownerId}")
    ApiResponse<List<PetResponse>> GetPetsByOwner(@PathVariable String ownerId) {
        return ApiResponse.<List<PetResponse>>builder()
                .result(petService.getPetsByOwner(ownerId))
                .build();
    }

    @PutMapping("/{PetId}")
    ApiResponse<PetResponse> UpdatePet(@PathVariable String PetId, @RequestBody PetUpdateRequest request) {
        return ApiResponse.<PetResponse>builder()
                .result(petService.updatePet(PetId, request))
                .build();
    }

    @DeleteMapping("/{petId}")
    ApiResponse<String> DeletePet(@PathVariable String petId) {
        return ApiResponse.<String>builder()
                .result(petService.deletePet(petId))
                .build();
    }
}
