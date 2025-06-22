package com.example.Backend_SmartVetSystem.service;

import com.example.Backend_SmartVetSystem.entity.Appointment;
import com.example.Backend_SmartVetSystem.repository.AppointmentRepository;
import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import org.quartz.*;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Date;
import java.util.List;

@Component
@RequiredArgsConstructor
public class AppointmentScheduler {

    private final AppointmentRepository appointmentRepository;
    private final Scheduler scheduler;

    public void UpdateSchedule(Appointment appointment) {
        JobKey jobKey = new JobKey(appointment.getAppointmentId(), "appointment");
        try {
            if(scheduler.checkExists(jobKey)){
                scheduler.deleteJob(jobKey);
            }
            createSchedule(appointment);
        } catch (SchedulerException e) {
            throw new RuntimeException("Error when create appointment", e);
        }
    }

    public void createSchedule(Appointment appointment) {
        Instant reminderTime = appointment.getAppointmentTime().minus(8, ChronoUnit.HOURS);

        if (reminderTime.isAfter(Instant.now())) {
            try {
                JobKey jobKey = new JobKey(appointment.getAppointmentId(), "appointment");

                if (!scheduler.checkExists(jobKey)) {
                    JobDetail jobDetail = JobBuilder.newJob(EmailReminderJob.class)
                            .withIdentity(jobKey)
                            .usingJobData("toEmail", appointment.getOwner().getEmail())
                            .usingJobData("appointmentTime", appointment.getAppointmentTime().toString())
                            .usingJobData("appointmentId", appointment.getAppointmentId())
                            .build();

                    Trigger trigger = TriggerBuilder.newTrigger()
                            .withIdentity(appointment.getAppointmentId(), "appointment")
                            .startAt(Date.from(reminderTime))
                            .build();

                    scheduler.scheduleJob(jobDetail, trigger);
                }

            } catch (SchedulerException e) {
                throw new RuntimeException("Error when create appointment", e);
            }
        }
    }
    @Transactional
    @EventListener(ApplicationReadyEvent.class)
    public void scheduleReminders() {
        List<Appointment> appointments = appointmentRepository.findByStatus("pending");
        for (Appointment appt : appointments) {
            createSchedule(appt);
        }
    }
}
