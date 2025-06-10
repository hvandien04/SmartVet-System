package com.example.Backend_SmartVetSystem.service;

import com.example.Backend_SmartVetSystem.dto.request.*;
import com.example.Backend_SmartVetSystem.dto.response.UserResponse;
import com.example.Backend_SmartVetSystem.entity.User;
import com.example.Backend_SmartVetSystem.exception.AppException;
import com.example.Backend_SmartVetSystem.exception.ErrorCode;
import com.example.Backend_SmartVetSystem.mapper.UserMapper;
import com.example.Backend_SmartVetSystem.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.prepost.PostAuthorize;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.AbstractMap;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor

public class UserService {

    private final UserRepository userRepository;
    private final UserMapper userMapper;
    private final PasswordEncoder passwordEncoder;
    private final IdGeneratorService idGeneratorService;
    private final EmailService emailService;
    private final ConcurrentHashMap<String, AbstractMap.SimpleEntry<String, LocalDateTime>> verificationCodes = new ConcurrentHashMap<>();


    @PreAuthorize("hasRole('ADMIN')")
    public UserResponse createUser(UserCreateRequest request) {
        User user = userMapper.toUser(request);
        user.setUserId(idGeneratorService.generateRandomId("U",userRepository::existsById));
        String defaultPassword = "12345678@aA";
        user.setPasswordHash(passwordEncoder.encode(defaultPassword));
        try {
            user = userRepository.save(user);
        } catch (Exception e) {
            throw new AppException(ErrorCode.USER_EXISTS);
        }
        return userMapper.toUserResponse(user);
    }

    @PreAuthorize("hasRole('ADMIN')")
    public UserResponse updateUser(String userId,UserUpdateRequest request) {
        User user = userRepository.findById(userId).orElseThrow(()-> new AppException(ErrorCode.USER_NOT_FOUND));
        userMapper.updateUser(user, request);
        user.setPasswordHash(passwordEncoder.encode(request.getPassword()));
        return userMapper.toUserResponse(userRepository.save(user));
    }

    @PostAuthorize("returnObject.username==authentication.name")
    public UserResponse updatePassword(UserUpdatePasswordRequest request) {
        var context = SecurityContextHolder.getContext();
        var username = context.getAuthentication().getName();
        User user = userRepository.findByUsername(username).orElseThrow(()-> new AppException(ErrorCode.USER_NOT_FOUND));
        if (!passwordEncoder.matches(request.getOldPassword(), user.getPasswordHash())) {
            throw new AppException(ErrorCode.OLD_PASSWORD_INCORRECT);
        }
        user.setPasswordHash(passwordEncoder.encode(request.getNewPassword()));
        return userMapper.toUserResponse(userRepository.save(user));
    }


    @PostAuthorize("returnObject.username==authentication.name")
    public UserResponse getMyInfo() {
        var context = SecurityContextHolder.getContext();
        var username = context.getAuthentication().getName();
        User user = userRepository.findByUsername(username).orElseThrow(()-> new AppException(ErrorCode.USER_NOT_FOUND));
        return userMapper.toUserResponse(user);
    }

    @PreAuthorize("hasRole('ADMIN')")
    public List<UserResponse> getAllUsers() {
        return userRepository.findAll().stream().map(userMapper::toUserResponse).collect(Collectors.toList());
    }

    private String generateVerificationCode() {
        int length = 6;
        String characters = "0123456789";
        Random random = new Random();
        StringBuilder code = new StringBuilder();

        for (int i = 0; i < length; i++) {
            code.append(characters.charAt(random.nextInt(characters.length())));
        }

        return code.toString();
    }

    public void generateVerificationCode(ForgotPasswordRequest request) throws IOException {
        userRepository.findByEmail(request.getEmail()).orElseThrow(()-> new AppException(ErrorCode.USER_NOT_FOUND));
        String code = generateVerificationCode();
        LocalDateTime expiry = LocalDateTime.now().plusMinutes(10);
        emailService.sendResetCode(request.getEmail(),code);
        verificationCodes.put(request.getEmail(), new AbstractMap.SimpleEntry<>(code, expiry));
    }

    public void resetPassword(String email, String newPassword) {
        Optional<User> optionalUser = userRepository.findByEmail(email);
        if (optionalUser.isPresent()) {
            User user = optionalUser.get();
            user.setPasswordHash(passwordEncoder.encode(newPassword));
            userRepository.save(user);
        }
    }

    public boolean verifyCode(String email, String code) {
        AbstractMap.SimpleEntry<String, LocalDateTime> entry = verificationCodes.get(email);
        if (entry == null) throw new AppException(ErrorCode.INVALID_CONFIRMATION_CODE);
        if (LocalDateTime.now().isAfter(entry.getValue())) {
            verificationCodes.remove(email);
            throw new AppException(ErrorCode.INVALID_CONFIRMATION_CODE);
        }
        if (!entry.getKey().equals(code)) {
            throw new AppException(ErrorCode.INVALID_CONFIRMATION_CODE);
        }
        return true;
    }

    public void clearCode(String email) {
        verificationCodes.remove(email);
    }

}
