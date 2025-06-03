package com.example.Backend_SmartVetSystem.controller;

import com.example.Backend_SmartVetSystem.dto.request.ApiResponse;
import com.example.Backend_SmartVetSystem.dto.request.AppointmentRequest;
import com.example.Backend_SmartVetSystem.dto.response.AppointmentResponse;
import com.example.Backend_SmartVetSystem.service.AppointmentService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/Appointment")
@RequiredArgsConstructor
public class AppointmentController {
    private final AppointmentService appointmentService;

    @PostMapping
    ApiResponse<AppointmentResponse> createAppointment(@RequestBody AppointmentRequest appointmentRequest) {
        return ApiResponse.<AppointmentResponse>builder()
                .result(appointmentService.createAppointment(appointmentRequest))
                .build();
    }

    @PutMapping("/{Id}")
    ApiResponse<AppointmentResponse> updateAppointment(@PathVariable String Id, @RequestBody AppointmentRequest appointmentRequest) {
        return ApiResponse.<AppointmentResponse>builder()
                .result(appointmentService.updateAppointment(Id,appointmentRequest))
                .build();
    }
}
