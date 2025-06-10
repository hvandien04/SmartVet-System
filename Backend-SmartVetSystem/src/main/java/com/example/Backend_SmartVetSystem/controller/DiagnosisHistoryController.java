package com.example.Backend_SmartVetSystem.controller;

import com.example.Backend_SmartVetSystem.dto.request.ApiResponse;
import com.example.Backend_SmartVetSystem.dto.request.DiagnosisHistoryRequest;
import com.example.Backend_SmartVetSystem.dto.response.DiagnosisHistoryResponse;
import com.example.Backend_SmartVetSystem.service.DiagnosisHistoryService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/diagnosis-history")
@RequiredArgsConstructor
public class DiagnosisHistoryController {
    private final DiagnosisHistoryService diagnosisHistoryService;

    @PostMapping
    ApiResponse<DiagnosisHistoryResponse> CreateDiagnosisHistory(@RequestBody DiagnosisHistoryRequest diagnosisHistoryRequest) {
        return ApiResponse.<DiagnosisHistoryResponse>builder()
                .result(diagnosisHistoryService.CreateDiagnosisHistory(diagnosisHistoryRequest))
                .build();
    }

    @GetMapping
    ApiResponse<List<DiagnosisHistoryResponse>> getDiagnosisHistory() {
        return ApiResponse.<List<DiagnosisHistoryResponse>>builder().result(diagnosisHistoryService.getAllDiagnosisHistory()).build();
    }

    @GetMapping("/recent")
    ApiResponse<List<DiagnosisHistoryResponse>> getRecentDiagnosisHistory() {
        return ApiResponse.<List<DiagnosisHistoryResponse>>builder()
                .result(diagnosisHistoryService.findTop10ByOrderByCreatedAtDesc())
                .build();
    }

}
