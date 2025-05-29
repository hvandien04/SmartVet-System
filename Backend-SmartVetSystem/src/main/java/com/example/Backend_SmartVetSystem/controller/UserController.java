package com.example.Backend_SmartVetSystem.controller;

import com.example.Backend_SmartVetSystem.dto.request.*;
import com.example.Backend_SmartVetSystem.dto.response.UserResponse;
import com.example.Backend_SmartVetSystem.service.UserService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;

@RestController
@RequiredArgsConstructor
@RequestMapping("/user")
public class UserController {

    final UserService userService;

    @PutMapping
    ApiResponse<UserResponse> UpdatePassword(@RequestBody UserUpdatePasswordRequest request) {
        return ApiResponse.<UserResponse>builder()
                .result(userService.updatePassword(request))
                .build();
    }

    @GetMapping
    ApiResponse<UserResponse> GetMyInfo() {
        return ApiResponse.<UserResponse>builder()
                .result(userService.getMyInfo())
                .build();
    }

    @PostMapping("/forgot-password")
    ApiResponse<Void> forgotPassword(@RequestBody ForgotPasswordRequest request) throws IOException {
        userService.generateVerificationCode(request);
        return ApiResponse.<Void>builder()
                .build();
    }

    @PostMapping("/verify-code")
    ApiResponse<Boolean> verifyCode(@RequestBody VerifyCodeRequest request) {
        var verified = userService.verifyCode(request.getEmail(),request.getCode());
        return ApiResponse.<Boolean>builder()
                .result(verified)
                .build();
    }

    @PostMapping("/reset-password")
    ApiResponse<Boolean> resetPassword(@RequestBody ResetPasswordRequest request) {
        var verified = userService.verifyCode(request.getEmail(),request.getCode());
        if (verified) {
            userService.resetPassword(request.getEmail(), request.getNewPassword());
            userService.clearCode(request.getEmail());
        }
        return ApiResponse.<Boolean>builder()
                .result(verified)
                .build();
    }
}
