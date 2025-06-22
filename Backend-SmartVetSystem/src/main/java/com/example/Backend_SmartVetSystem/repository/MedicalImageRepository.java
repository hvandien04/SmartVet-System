package com.example.Backend_SmartVetSystem.repository;

import com.example.Backend_SmartVetSystem.entity.MedicalImage;
import com.example.Backend_SmartVetSystem.entity.MedicalRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface MedicalImageRepository extends JpaRepository<MedicalImage, String> {
    void deleteAllByRecord(MedicalRecord record);
}
