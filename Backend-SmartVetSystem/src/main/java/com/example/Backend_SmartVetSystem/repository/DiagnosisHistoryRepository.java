package com.example.Backend_SmartVetSystem.repository;

import com.example.Backend_SmartVetSystem.entity.DiagnosisHistory;
import com.example.Backend_SmartVetSystem.entity.MedicalRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface DiagnosisHistoryRepository extends JpaRepository<DiagnosisHistory, String> {
    List<DiagnosisHistory> findTop10ByOrderByCreatedAtDesc();

    void deleteAllByRecord(MedicalRecord medicalRecord);
}
