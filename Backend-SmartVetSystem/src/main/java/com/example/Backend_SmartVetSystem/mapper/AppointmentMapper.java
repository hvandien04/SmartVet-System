package com.example.Backend_SmartVetSystem.mapper;

import com.example.Backend_SmartVetSystem.dto.request.AppointmentRequest;
import com.example.Backend_SmartVetSystem.dto.response.AppointmentResponse;
import com.example.Backend_SmartVetSystem.entity.Appointment;
import org.mapstruct.Mapper;
import org.mapstruct.MappingTarget;

@Mapper(componentModel = "spring")
public interface AppointmentMapper {

    Appointment toAppointment(AppointmentRequest appointmentRequest);
    AppointmentResponse toAppointmentResponse(Appointment appointment);

    void UpdateAppointment(@MappingTarget Appointment appointment, AppointmentRequest appointmentRequest);
}
