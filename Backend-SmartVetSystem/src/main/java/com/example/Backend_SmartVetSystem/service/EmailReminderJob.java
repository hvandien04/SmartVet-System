package com.example.Backend_SmartVetSystem.service;

import lombok.RequiredArgsConstructor;
import org.quartz.Job;
import org.quartz.JobDataMap;
import org.quartz.JobExecutionException;
import org.springframework.stereotype.Component;

import java.io.IOException;

@Component
@RequiredArgsConstructor
public class EmailReminderJob implements Job {

    private final EmailService emailService;
    private final NotificationLogService notificationLogService;

    @Override
    public void execute(org.quartz.JobExecutionContext context) throws JobExecutionException {
        JobDataMap dataMap = context.getMergedJobDataMap();
        String toEmail = dataMap.getString("toEmail");
        String appointmentTime = dataMap.getString("appointmentTime");
        try {
            emailService.sendAppointmentReminder(toEmail, appointmentTime);
            notificationLogService.saveNotificationLog(dataMap.getString("appointmentId"), toEmail, "sent");
        } catch (IOException e) {
            notificationLogService.saveNotificationLog(dataMap.getString("appointmentId"), toEmail, "failed");
            throw new JobExecutionException(e);
        }
    }
}
