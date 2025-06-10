package com.example.Backend_SmartVetSystem.repository;

import com.example.Backend_SmartVetSystem.entity.DiseasePrediction;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface DiseasePredictionRepository extends JpaRepository<DiseasePrediction, String> {
}
