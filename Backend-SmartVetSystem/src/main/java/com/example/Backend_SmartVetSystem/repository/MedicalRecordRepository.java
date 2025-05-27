package com.example.Backend_SmartVetSystem.repository;

import com.example.Backend_SmartVetSystem.entity.MedicalRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface MedicalRecordRepository extends JpaRepository<MedicalRecord, String> {
}
