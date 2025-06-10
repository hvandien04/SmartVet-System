package com.example.Backend_SmartVetSystem.exception;

import lombok.Getter;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;

@Getter
public enum ErrorCode {
    UNCATEGORIZED_EXCEPTION(9999, "Uncategorized error", HttpStatus.INTERNAL_SERVER_ERROR),
    UNAUTHORIZED(1000, "You do not have permission", HttpStatus.FORBIDDEN),
    INVALID_REFRESH_TOKEN(1001, "Invalid refresh token" , HttpStatus.BAD_REQUEST ),
    USER_EXISTS(1002, "Username or email already exists", HttpStatus.CONFLICT),
    USER_NOT_FOUND(1003, "User not found", HttpStatus.NOT_FOUND),
    OLD_PASSWORD_INCORRECT(1004, "Old password incorrect", HttpStatus.BAD_REQUEST),
    INVALID_PASSWORD(1005, "Invalid password", HttpStatus.BAD_REQUEST),
    UNAUTHENTICATED(1006, "Unauthenticated", HttpStatus.UNAUTHORIZED),
    OWNER_NOT_FOUND(1007, "Owner not found", HttpStatus.NOT_FOUND),
    PET_NOT_FOUND(1008, "Pet not found", HttpStatus.NOT_FOUND),
    MEDICAL_RECORD_NOT_FOUND(1009, "Medical record not found", HttpStatus.NOT_FOUND),
    INVALID_CONFIRMATION_CODE(1010, "The confirmation code is incorrect or has expired.", HttpStatus.NOT_FOUND),
    DIAGNOSTIC_HISTORY_NOT_FOUND(1011, "Diagnostic history not found", HttpStatus.NOT_FOUND),
    APPLICATION_NOT_FOUND(1012, "Application not found", HttpStatus.NOT_FOUND),
    HAS_ANOTHER_APPOINTMENTS(1013, "This doctor has another appointment.", HttpStatus.NOT_FOUND),
    DISEASE_PREDICTION_NOT_FOUND(1013,"Disease prediction not found", HttpStatus.NOT_FOUND),
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
