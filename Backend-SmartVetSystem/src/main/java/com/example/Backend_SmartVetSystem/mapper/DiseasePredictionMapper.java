package com.example.Backend_SmartVetSystem.mapper;


import com.example.Backend_SmartVetSystem.dto.request.DiseasePredictionRequest;
import com.example.Backend_SmartVetSystem.dto.response.DiseasePredictionResponse;
import com.example.Backend_SmartVetSystem.entity.DiseasePrediction;
import org.mapstruct.Mapper;
import org.mapstruct.MappingTarget;

@Mapper(componentModel = "spring")
public interface DiseasePredictionMapper {
    DiseasePrediction toDiseasePrediction(DiseasePredictionRequest request);
    DiseasePredictionResponse toDiseasePredictionResponse(DiseasePrediction diseasePrediction);

    void UpdateDiseasePrediction (@MappingTarget DiseasePrediction diseasePrediction, DiseasePredictionRequest request);

}
