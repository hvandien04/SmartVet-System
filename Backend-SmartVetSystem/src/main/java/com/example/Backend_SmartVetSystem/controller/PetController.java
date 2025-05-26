package com.example.Backend_SmartVetSystem.controller;

import com.example.Backend_SmartVetSystem.dto.request.ApiResponse;
import com.example.Backend_SmartVetSystem.dto.request.PetCreateRequest;
import com.example.Backend_SmartVetSystem.dto.response.PetResponse;
import com.example.Backend_SmartVetSystem.service.PetService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

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
}
