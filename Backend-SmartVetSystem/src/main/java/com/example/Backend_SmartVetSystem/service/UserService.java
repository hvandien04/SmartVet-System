package com.example.Backend_SmartVetSystem.service;

import com.example.Backend_SmartVetSystem.dto.request.UserCreateRequest;
import com.example.Backend_SmartVetSystem.dto.request.UserUpdatePasswordRequest;
import com.example.Backend_SmartVetSystem.dto.request.UserUpdateRequest;
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

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor

public class UserService {

    private final UserRepository userRepository;
    private final UserMapper userMapper;
    private final PasswordEncoder passwordEncoder;
    private final IdGeneratorService idGeneratorService;

    @PreAuthorize("hasRole('ADMIN')")
    public UserResponse createUser(UserCreateRequest request) {
        User user = userMapper.toUser(request);
        user.setUserId(idGeneratorService.generateRandomId("U",userRepository::existsById));
        user.setPasswordHash(passwordEncoder.encode(request.getPassword()));
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

    @PreAuthorize("hasRole('ADMIN')")
    public List<UserResponse> getAllUsers() {
        return userRepository.findAll().stream().map(userMapper::toUserResponse).collect(Collectors.toList());
    }


}
