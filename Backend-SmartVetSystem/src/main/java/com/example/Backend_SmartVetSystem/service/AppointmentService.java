package com.example.Backend_SmartVetSystem.service;


import com.example.Backend_SmartVetSystem.dto.request.AppointmentRequest;
import com.example.Backend_SmartVetSystem.dto.response.AppointmentResponse;
import com.example.Backend_SmartVetSystem.entity.Appointment;
import com.example.Backend_SmartVetSystem.entity.Owner;
import com.example.Backend_SmartVetSystem.entity.User;
import com.example.Backend_SmartVetSystem.exception.AppException;
import com.example.Backend_SmartVetSystem.exception.ErrorCode;
import com.example.Backend_SmartVetSystem.mapper.AppointmentMapper;
import com.example.Backend_SmartVetSystem.repository.AppointmentRepository;
import com.example.Backend_SmartVetSystem.repository.OwnerRepository;
import com.example.Backend_SmartVetSystem.repository.PetRepository;
import com.example.Backend_SmartVetSystem.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class AppointmentService {
    private final AppointmentRepository appointmentRepository;
    private final AppointmentMapper appointmentMapper;
    private final IdGeneratorService idGeneratorService;
    private final PetRepository petRepository;
    private final OwnerRepository ownerRepository;
    private final UserRepository userRepository;
    private final AppointmentScheduler appointmentScheduler;

    private boolean findConflictAppointment(String userId, Instant newTime) {
        List<Appointment> conflicts = appointmentRepository.findConflictAppointment(userId, newTime.minus(1, ChronoUnit.HOURS), newTime.plus(1, ChronoUnit.MINUTES) );
        return !conflicts.isEmpty();
    }

    private boolean findUpdateAppointment(String userId, Instant newTime, String appointmentId) {
        List<Appointment> conflicts = appointmentRepository.findUpdateAppointments(userId, newTime.minus(1, ChronoUnit.HOURS), newTime.plus(1, ChronoUnit.MINUTES), appointmentId);
        return !conflicts.isEmpty();
    }

    public AppointmentResponse createAppointment(AppointmentRequest appointmentRequest) {
        Appointment appointment = appointmentMapper.toAppointment(appointmentRequest);
        appointment.setAppointmentId(idGeneratorService.generateRandomId("A",appointmentRepository::existsById));
        petRepository.findById(appointmentRequest.getPetId()).orElseThrow(()-> new AppException(ErrorCode.PET_NOT_FOUND));
        Owner owner = ownerRepository.findById(appointmentRequest.getOwnerId()).orElseThrow(()-> new AppException(ErrorCode.OWNER_NOT_FOUND));
        User  user = userRepository.findById(appointmentRequest.getUserId()).orElseThrow(()-> new AppException(ErrorCode.USER_NOT_FOUND));
        if(findConflictAppointment(user.getUserId(),appointmentRequest.getAppointmentTime())){
            throw new AppException(ErrorCode.HAS_ANOTHER_APPOINTMENTS);
        }
        appointment.setOwner(owner);
        appointment.setUser(user);
        appointmentScheduler.createSchedule( appointment);
        return appointmentMapper.toAppointmentResponse(appointmentRepository.save(appointment));
    }

    public AppointmentResponse getAppointment(String appointmentId) {
        return appointmentMapper.toAppointmentResponse(appointmentRepository.findById(appointmentId).orElseThrow(() -> new AppException(ErrorCode.APPLICATION_NOT_FOUND)));
    }

    public List<AppointmentResponse> getAllAppointments() {
        return appointmentRepository.findAll().stream().map(appointmentMapper::toAppointmentResponse).collect(Collectors.toList());
    }

    public List<AppointmentResponse> getAppointmentsByInstant(Instant startDate, Instant endDate) {
        return appointmentRepository.appointmentByTime(startDate, endDate).stream().map(appointmentMapper::toAppointmentResponse).collect(Collectors.toList());
    }

    public AppointmentResponse updateAppointment(String appointmentId, AppointmentRequest appointmentRequest) {
        Appointment appointment = appointmentRepository.findById(appointmentId).orElseThrow(() -> new AppException(ErrorCode.APPLICATION_NOT_FOUND));
        if(appointmentRequest.getAppointmentTime() != null) {
            if(findUpdateAppointment(appointment.getUser().getUserId(),appointmentRequest.getAppointmentTime(),appointmentId)){
                throw new AppException(ErrorCode.HAS_ANOTHER_APPOINTMENTS);
            }
        }
        appointmentMapper.UpdateAppointment(appointment, appointmentRequest);
        petRepository.findById(appointmentRequest.getPetId()).orElseThrow(()-> new AppException(ErrorCode.PET_NOT_FOUND));
        Owner owner = ownerRepository.findById(appointmentRequest.getOwnerId()).orElseThrow(()-> new AppException(ErrorCode.OWNER_NOT_FOUND));
        User  user = userRepository.findById(appointmentRequest.getUserId()).orElseThrow(()-> new AppException(ErrorCode.USER_NOT_FOUND));
        appointment.setOwner(owner);
        appointment.setUser(user);
        if (appointmentRequest.getAppointmentTime() != null &&
                !appointmentRequest.getAppointmentTime().equals(appointment.getAppointmentTime())) {
            appointmentScheduler.UpdateSchedule(appointment);
        }
        return appointmentMapper.toAppointmentResponse(appointmentRepository.save(appointment));
    }

    public String deleteAppointment(String appointmentId) {
        appointmentRepository.deleteById(appointmentId);
        return "Appointment Id" + appointmentId + " deleted";
    }
}
