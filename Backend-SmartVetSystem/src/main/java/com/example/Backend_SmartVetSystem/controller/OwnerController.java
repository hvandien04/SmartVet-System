package com.example.Backend_SmartVetSystem.controller;

import com.example.Backend_SmartVetSystem.dto.request.ApiResponse;
import com.example.Backend_SmartVetSystem.dto.request.OwnerRequest;
import com.example.Backend_SmartVetSystem.dto.response.OwnerResponse;
import com.example.Backend_SmartVetSystem.service.OwnerService;
import lombok.Builder;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("owner")
public class OwnerController {
    private final OwnerService ownerService;

    @PostMapping
    ApiResponse<OwnerResponse> createOwner(@RequestBody OwnerRequest ownerRequest) {
        return ApiResponse.<OwnerResponse>builder()
                .result(ownerService.createOwner(ownerRequest))
                .build();
    }

    @PutMapping("/{OwnerId}")
    ApiResponse<OwnerResponse> updateOwner(@PathVariable String OwnerId,@RequestBody OwnerRequest ownerRequest) {
        return ApiResponse.<OwnerResponse>builder()
                .result(ownerService.updateOwner(OwnerId,ownerRequest))
                .build();
    }

    @GetMapping
    ApiResponse<List<OwnerResponse>> getAllOwner() {
        return ApiResponse.<List<OwnerResponse>>builder()
                .result(ownerService.getAllOwners())
                .build();
    }

    @GetMapping("/{ownerId}")
    ApiResponse<OwnerResponse> getOwner(@PathVariable String ownerId) {
        return ApiResponse.<OwnerResponse>builder()
                .result(ownerService.getOwner(ownerId))
                .build();
    }

    @DeleteMapping("/{ownerId}")
    ApiResponse<String> deleteOwner(@PathVariable String ownerId) {
        return ApiResponse.<String>builder()
                .result(ownerService.deleteOwner(ownerId))
                .build();
    }
}
