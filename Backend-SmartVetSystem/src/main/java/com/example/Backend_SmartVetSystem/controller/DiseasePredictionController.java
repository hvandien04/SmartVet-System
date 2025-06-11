package com.example.Backend_SmartVetSystem.controller;

import com.example.Backend_SmartVetSystem.dto.request.ApiResponse;
import com.example.Backend_SmartVetSystem.dto.request.DiseasePredictionRequest;
import com.example.Backend_SmartVetSystem.dto.response.DiseasePredictionResponse;
import com.example.Backend_SmartVetSystem.service.DiseasePredictionService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.util.List;
import java.util.Locale;

@RestController
@RequiredArgsConstructor
@RequestMapping("disease-predict")
public class DiseasePredictionController {
    private final DiseasePredictionService diseasePredictionService;

    @PostMapping
    ApiResponse<DiseasePredictionResponse> createDiseasePrediction(@RequestBody DiseasePredictionRequest request) {
        return ApiResponse.<DiseasePredictionResponse>builder()
                .result(diseasePredictionService.createDiseasePrediction(request))
                .build();
    }

    @PutMapping("/{Id}")
    ApiResponse<DiseasePredictionResponse> updateDiseasePrediction(@PathVariable String Id, @RequestBody DiseasePredictionRequest request) {
        return ApiResponse.<DiseasePredictionResponse>builder()
                .result(diseasePredictionService.updateDiseasePrediction(Id, request))
                .build();
    }

    @GetMapping
    ApiResponse<List<DiseasePredictionResponse>> getAlL(){
        return ApiResponse.<List<DiseasePredictionResponse>>builder()
                .result(diseasePredictionService.getAllDiseasePrediction())
                .build();
    }

    @GetMapping("/{Id}")
    ApiResponse<DiseasePredictionResponse> getDiseasePrediction(@PathVariable String Id){
        return ApiResponse.<DiseasePredictionResponse>builder()
                .result(diseasePredictionService.findDiseasePrediction(Id))
                .build();
    }

    @GetMapping("/my-disease-predict")
    ApiResponse<List<DiseasePredictionResponse>> getMyPrediction(){
        return ApiResponse.<List<DiseasePredictionResponse>>builder()
                .result(diseasePredictionService.getMyDiseasePrediction())
                .build();
    }

    @GetMapping("/my-disease-predict/{date}")
    ApiResponse<List<DiseasePredictionResponse>> getMyPredictionByDate(@PathVariable String date){
        LocalDate date1 = LocalDate.parse(date);
        return ApiResponse.<List<DiseasePredictionResponse>>builder()
                .result(diseasePredictionService.getDiseasePredictionByTime(date1))
                .build();
    }

    @DeleteMapping("/{Id}")
    ApiResponse<String> Delete(@PathVariable String Id){
        return ApiResponse.<String>builder()
                .result(diseasePredictionService.deleteDiseasePrediction(Id))
                .build();
    }
}
