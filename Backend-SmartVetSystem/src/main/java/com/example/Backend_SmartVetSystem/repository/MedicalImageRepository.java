package com.example.Backend_SmartVetSystem.repository;

import com.example.Backend_SmartVetSystem.entity.MedicalImage;
import org.springframework.data.jpa.repository.JpaRepository;

public interface MedicalImageRepository extends JpaRepository<MedicalImage, String> {
}
