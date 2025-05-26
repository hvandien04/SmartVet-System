package com.example.Backend_SmartVetSystem.mapper;

import com.example.Backend_SmartVetSystem.dto.request.UserCreateRequest;
import com.example.Backend_SmartVetSystem.dto.request.UserUpdateRequest;
import com.example.Backend_SmartVetSystem.dto.response.UserResponse;
import com.example.Backend_SmartVetSystem.entity.User;
import org.mapstruct.Mapper;
import org.mapstruct.MappingTarget;

@Mapper(componentModel = "spring")
public interface UserMapper {
    User toUser(UserCreateRequest request);

    UserResponse toUserResponse(User user);

    void updateUser(@MappingTarget User user, UserUpdateRequest request);
}
