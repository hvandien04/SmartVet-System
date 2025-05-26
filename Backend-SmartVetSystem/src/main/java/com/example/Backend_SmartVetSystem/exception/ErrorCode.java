package com.example.Backend_SmartVetSystem.exception;

import lombok.Getter;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;

@Getter
public enum ErrorCode {
    UNCATEGORIZED_EXCEPTION(9999, "Uncategorized error", HttpStatus.INTERNAL_SERVER_ERROR),
    UNAUTHORIZED(1000, "You do not have permission", HttpStatus.FORBIDDEN),
    INVALID_REFRESH_TOKEN(1001, "Invalid refresh token" , HttpStatus.BAD_REQUEST ),
    USER_EXISTS(1002, "User already exists", HttpStatus.CONFLICT),
    USER_NOT_FOUND(1003, "User not found", HttpStatus.NOT_FOUND),
    OLD_PASSWORD_INCORRECT(1004, "Old password incorrect", HttpStatus.BAD_REQUEST),
    INVALID_PASSWORD(1005, "Invalid password", HttpStatus.BAD_REQUEST),
    UNAUTHENTICATED(1006, "Unauthenticated", HttpStatus.UNAUTHORIZED),
    OWNER_NOT_FOUND(1007, "Owner not found", HttpStatus.NOT_FOUND),
    ;
    ErrorCode(int code, String message, HttpStatusCode statusCode) {
        this.code = code;
        this.message = message;
        this.statusCode = statusCode;
    }

    private final int code;
    private final String message;
    private final HttpStatusCode statusCode;
}
