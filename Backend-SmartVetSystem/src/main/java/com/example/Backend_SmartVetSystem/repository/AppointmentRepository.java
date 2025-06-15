package com.example.Backend_SmartVetSystem.repository;

import com.example.Backend_SmartVetSystem.entity.Appointment;
import com.example.Backend_SmartVetSystem.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.Instant;
import java.util.List;

@Repository
public interface AppointmentRepository extends JpaRepository<Appointment, String> {
    @Query("SELECT a FROM Appointment a " +
            "WHERE a.user.userId = :userId " +
            "AND a.appointmentTime BETWEEN :startTime AND :endTime ")
    List<Appointment> findConflictAppointment(
            @Param("userId") String userId,
            @Param("startTime") Instant startTime,
            @Param("endTime") Instant endTime
    );

    @Query("SELECT a FROM Appointment a " +
            "WHERE a.user.userId = :userId " +
            "AND a.appointmentTime BETWEEN :startTime AND :endTime " +
            "AND a.appointmentId <> :excludeId")
    List<Appointment> findUpdateAppointments(
            @Param("userId") String userId,
            @Param("startTime") Instant startTime,
            @Param("endTime") Instant endTime,
            @Param("excludeId") String excludeId
    );

    @Query("SELECT a FROM Appointment a " +
            "WHERE a.appointmentTime BETWEEN :startDate AND :endDate")
    List<Appointment> appointmentByTime(
            @Param("startDate") Instant startDate,
            @Param("endDate") Instant endDate
    );

    List<Appointment> findByStatus(String pending);
}
