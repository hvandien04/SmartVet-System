package com.example.Backend_SmartVetSystem.controller;

import com.example.Backend_SmartVetSystem.dto.request.ApiResponse;
import com.example.Backend_SmartVetSystem.dto.request.UserUpdatePasswordRequest;
import com.example.Backend_SmartVetSystem.dto.response.UserResponse;
import com.example.Backend_SmartVetSystem.service.UserService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

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
}
