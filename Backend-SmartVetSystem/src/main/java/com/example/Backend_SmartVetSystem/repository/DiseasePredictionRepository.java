package com.example.Backend_SmartVetSystem.repository;

import com.example.Backend_SmartVetSystem.entity.DiseasePrediction;
import com.example.Backend_SmartVetSystem.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;

@Repository
public interface DiseasePredictionRepository extends JpaRepository<DiseasePrediction, String> {
    List<DiseasePrediction> findAllByUser(User user);


    @Query("SELECT a FROM DiseasePrediction a WHERE DATE(a.createdAt) = :date AND a.user.userId = :userId ")
    List<DiseasePrediction> findDiseasePredictionByDate(@Param("userId")String UserId ,@Param("date") LocalDate date);
}
