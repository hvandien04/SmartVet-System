package com.example.Backend_SmartVetSystem.service;

import com.example.Backend_SmartVetSystem.entity.Appointment;
import com.example.Backend_SmartVetSystem.entity.NotificationLog;
import com.example.Backend_SmartVetSystem.repository.AppointmentRepository;
import com.example.Backend_SmartVetSystem.repository.NotificationLogRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.time.temporal.ChronoUnit;

@Service
@RequiredArgsConstructor
public class NotificationLogService {

    private final NotificationLogRepository notificationLogRepository;
    private final IdGeneratorService idGeneratorService;
    private final AppointmentRepository appointmentRepository;

    public void saveNotificationLog(String appointmentId, String emailTo, String status){
        NotificationLog notificationLog = new NotificationLog();
        Appointment appointment = appointmentRepository.findById(appointmentId).orElseThrow(()-> new RuntimeException("Appointment not found"));
        notificationLog.setAppointment(appointment);
        notificationLog.setEmailTo(emailTo);
        notificationLog.setStatus(status);
        notificationLog.setSentTime(Instant.now().plus(7, ChronoUnit.HOURS));
        notificationLog.setNotificationId(idGeneratorService.generateRandomId("NL",notificationLogRepository::existsById));
        notificationLogRepository.save(notificationLog);
    }
}
