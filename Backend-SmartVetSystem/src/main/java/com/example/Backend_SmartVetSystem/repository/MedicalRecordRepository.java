package com.example.Backend_SmartVetSystem.repository;

import com.example.Backend_SmartVetSystem.entity.MedicalRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.time.Instant;
import java.util.List;

@Repository
public interface MedicalRecordRepository extends JpaRepository<MedicalRecord, String> {
    List<MedicalRecord> findByVisitDate(Instant visitDate);
}
