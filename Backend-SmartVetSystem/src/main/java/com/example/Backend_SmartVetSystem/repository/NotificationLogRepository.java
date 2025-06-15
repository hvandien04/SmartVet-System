package com.example.Backend_SmartVetSystem.repository;

import com.example.Backend_SmartVetSystem.entity.NotificationLog;
import org.springframework.data.jpa.repository.JpaRepository;

public interface NotificationLogRepository extends JpaRepository<NotificationLog, String> {
}
