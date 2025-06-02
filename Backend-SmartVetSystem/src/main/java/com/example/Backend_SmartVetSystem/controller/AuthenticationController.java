package com.example.Backend_SmartVetSystem.controller;

import com.example.Backend_SmartVetSystem.dto.request.ApiResponse;
import com.example.Backend_SmartVetSystem.dto.request.AuthenticationRequest;
import com.example.Backend_SmartVetSystem.dto.request.IntrospectRequest;
import com.example.Backend_SmartVetSystem.dto.request.UserUpdatePasswordRequest;
import com.example.Backend_SmartVetSystem.dto.response.AuthenticationResponse;
import com.example.Backend_SmartVetSystem.dto.response.UserResponse;
import com.example.Backend_SmartVetSystem.exception.AppException;
import com.example.Backend_SmartVetSystem.exception.ErrorCode;
import com.example.Backend_SmartVetSystem.service.AuthenticationService;
import com.example.Backend_SmartVetSystem.service.UserService;
import com.nimbusds.jose.JOSEException;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseCookie;
import org.springframework.web.bind.annotation.*;

import java.text.ParseException;

@RequiredArgsConstructor
@RestController
@RequestMapping("/auth")
public class AuthenticationController {
    private final AuthenticationService authenticationService;
    private final UserService userService;

    @PostMapping
    public ApiResponse<AuthenticationResponse> login(@RequestBody AuthenticationRequest request, HttpServletResponse response) {
        AuthenticationResponse authenticationResponse = authenticationService.login(request);
        ResponseCookie refreshCookie = ResponseCookie.from("refresh_token",authenticationResponse.getRefreshToken())
                .secure(false)
                .httpOnly(true)
                .maxAge(30*24*60*60)
                .path("/")
                .build();
        response.setHeader(HttpHeaders.SET_COOKIE, refreshCookie.toString());

        return ApiResponse.<AuthenticationResponse>builder()
                .result(authenticationResponse)
                .build();
    }

    @PostMapping("/logout")
    ApiResponse<Void> logout(@RequestBody IntrospectRequest request,
                             @CookieValue(name = "refresh_token", required = false) String refreshToken,
                             HttpServletResponse response) throws ParseException, JOSEException {
        System.out.println("Received logout request with body: " + request);
        System.out.println("Received refresh token from cookie: " + refreshToken);

        // Gọi service logout
        authenticationService.logout(request, refreshToken);

        // Xóa cookie refresh_token
        ResponseCookie clearCookie = ResponseCookie.from("refresh_token", "")
                .secure(false)
                .httpOnly(true)
                .maxAge(0)
                .path("/")
                .build();
        response.setHeader(HttpHeaders.SET_COOKIE, clearCookie.toString());

        return ApiResponse.<Void>builder().build();
    }


    @PostMapping("/refresh-token")
    public ApiResponse<AuthenticationResponse> refreshToken(
            @CookieValue(name = "refresh_token", required = false) String refreshToken) throws ParseException, JOSEException {
        System.out.println("Refresh token from cookie: " + refreshToken);
        if (refreshToken == null) {
            throw new AppException(ErrorCode.INVALID_REFRESH_TOKEN);
        }
        var newAccessToken = authenticationService.refreshAccessToken(refreshToken);
        return ApiResponse.<AuthenticationResponse>builder()
                .result(newAccessToken)
                .build();
    }


}
