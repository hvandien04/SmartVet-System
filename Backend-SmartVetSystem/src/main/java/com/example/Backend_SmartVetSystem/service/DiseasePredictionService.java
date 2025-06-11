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
import org.springframework.security.access.prepost.PostFilter;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
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
        diseasePrediction.setCreatedAt(Instant.now().plus(7, ChronoUnit.HOURS));
        return diagnosePredictionMapper.toDiseasePredictionResponse(diseasePredictionRepository.save(diseasePrediction));
    }

    @PreAuthorize("hasRole('ADMIN')")
    public DiseasePredictionResponse updateDiseasePrediction(String Id,DiseasePredictionRequest request){
        DiseasePrediction diseasePrediction = diseasePredictionRepository.findById(Id).orElseThrow(()-> new AppException(ErrorCode.DISEASE_PREDICTION_NOT_FOUND));
        diagnosePredictionMapper.UpdateDiseasePrediction(diseasePrediction,request);
        User user = userRepository.findById(request.getUserId()).orElseThrow(()-> new AppException(ErrorCode.USER_NOT_FOUND));
        diseasePrediction.setUser(user);
        return diagnosePredictionMapper.toDiseasePredictionResponse(diseasePredictionRepository.save(diseasePrediction));
    }


    public DiseasePredictionResponse findDiseasePrediction(String id){
        DiseasePrediction diseasePrediction = diseasePredictionRepository.findById(id).orElseThrow(()-> new AppException(ErrorCode.DISEASE_PREDICTION_NOT_FOUND));
        return diagnosePredictionMapper.toDiseasePredictionResponse(diseasePrediction);
    }


    @PreAuthorize("hasRole('ADMIN')")
    public List<DiseasePredictionResponse> getAllDiseasePrediction(){
        return diseasePredictionRepository.findAll().stream().map(diagnosePredictionMapper::toDiseasePredictionResponse).collect(Collectors.toList());
    }

    @PostFilter("filterObject.user.username == authentication.name")
    public List<DiseasePredictionResponse> getDiseasePredictionByTime(LocalDate time) {
        var username = SecurityContextHolder.getContext().getAuthentication().getName();
        User user = userRepository.findByUsername(username).orElseThrow(()-> new AppException(ErrorCode.USER_NOT_FOUND));
        return diseasePredictionRepository.findDiseasePredictionByDate(user.getUserId() ,time).stream().map(diagnosePredictionMapper::toDiseasePredictionResponse).collect(Collectors.toList());
    }

    public List<DiseasePredictionResponse> getMyDiseasePrediction(){
        var username = SecurityContextHolder.getContext().getAuthentication().getName();
        User user = userRepository.findByUsername(username).orElseThrow(()-> new AppException(ErrorCode.USER_NOT_FOUND));
        return diseasePredictionRepository.findAllByUser(user).stream().map(diagnosePredictionMapper::toDiseasePredictionResponse).collect(Collectors.toList());
    }

    @PreAuthorize("hasRole('ADMIN')")
    public String deleteDiseasePrediction(String id){
        diseasePredictionRepository.deleteById(id);
        return "Disease Prediction has Id " + id + " deleted";
    }
}
