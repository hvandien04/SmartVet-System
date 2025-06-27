package com.example.Backend_SmartVetSystem.repository;

import com.example.Backend_SmartVetSystem.entity.MedicalRecord;
import com.example.Backend_SmartVetSystem.entity.Pet;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;

@Repository
public interface MedicalRecordRepository extends JpaRepository<MedicalRecord, String> {

    @Query("SELECT a FROM MedicalRecord a WHERE DATE(a.visitDate) = :visitDate")
    List<MedicalRecord> findByVisitDate(@Param("visitDate") LocalDate visitDate);

    List<MedicalRecord> findByPet(Pet pet);
}
