package com.example.Backend_SmartVetSystem.controller;

import com.example.Backend_SmartVetSystem.dto.request.ApiResponse;
import com.example.Backend_SmartVetSystem.dto.request.AppointmentRequest;
import com.example.Backend_SmartVetSystem.dto.response.AppointmentResponse;
import com.example.Backend_SmartVetSystem.service.AppointmentService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.time.Instant;
import java.util.List;

@RestController
@RequestMapping("/appointment")
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

    @GetMapping
    ApiResponse<List<AppointmentResponse>> getAllAppointments() {
        return ApiResponse.<List<AppointmentResponse>>builder()
                .result(appointmentService.getAllAppointments())
                .build();
    }

    @GetMapping("/{Id}")
    ApiResponse<AppointmentResponse> getAppointment(@PathVariable String Id) {
        return ApiResponse.<AppointmentResponse>builder()
                .result(appointmentService.getAppointment(Id))
                .build();
    }

    @GetMapping("/by-time")
    ApiResponse<List<AppointmentResponse>> getAppointmentsByTime(@RequestParam("from") Instant from, @RequestParam("to") Instant to) {
        return ApiResponse.<List<AppointmentResponse>>builder()
                .result(appointmentService.getAppointmentsByInstant(from, to))
                .build();
    }

    @DeleteMapping("/{Id}")
    ApiResponse<String> deleteAppointment(@PathVariable String Id) {
        return ApiResponse.<String>builder()
                .result(appointmentService.deleteAppointment(Id))
                .build();
    }
}
