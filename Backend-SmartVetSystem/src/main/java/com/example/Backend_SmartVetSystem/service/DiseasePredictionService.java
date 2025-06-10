package com.example.Backend_SmartVetSystem.service;

import com.example.Backend_SmartVetSystem.dto.request.DiseasePredictionRequest;
import com.example.Backend_SmartVetSystem.dto.response.DiseasePredictionResponse;
import com.example.Backend_SmartVetSystem.entity.DiseasePrediction;
import com.example.Backend_SmartVetSystem.entity.User;
import com.example.Backend_SmartVetSystem.exception.AppException;
import com.example.Backend_SmartVetSystem.exception.ErrorCode;
import com.example.Backend_SmartVetSystem.mapper.DiseasePredictionMapper;
import com.example.Backend_SmartVetSystem.repository.DiseasePredictionRepository;
import com.example.Backend_SmartVetSystem.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class DiseasePredictionService {
    private final DiseasePredictionRepository diseasePredictionRepository;
    private final DiseasePredictionMapper diagnosePredictionMapper;
    private final IdGeneratorService idGeneratorService;
    private final UserRepository userRepository;

    public DiseasePredictionResponse createDiseasePrediction(DiseasePredictionRequest request){
        DiseasePrediction diseasePrediction = diagnosePredictionMapper.toDiseasePrediction(request);
        User user = userRepository.findById(request.getUserId()).orElseThrow(()-> new AppException(ErrorCode.USER_NOT_FOUND));
        diseasePrediction.setUser(user);
        diseasePrediction.setPredictionId(idGeneratorService.generateRandomId("dP",diseasePredictionRepository::existsById));
        return diagnosePredictionMapper.toDiseasePredictionResponse(diseasePredictionRepository.save(diseasePrediction));
    }

    public DiseasePredictionResponse findDiseasePrediction(String id){
        DiseasePrediction diseasePrediction = diseasePredictionRepository.findById(id).orElseThrow(()-> new AppException(ErrorCode.DISEASE_PREDICTION_NOT_FOUND));
        return diagnosePredictionMapper.toDiseasePredictionResponse(diseasePrediction);
    }

    public List<DiseasePredictionResponse> getAllDiseasePrediction(){
        return diseasePredictionRepository.findAll().stream().map(diagnosePredictionMapper::toDiseasePredictionResponse).collect(Collectors.toList());
    }
}
