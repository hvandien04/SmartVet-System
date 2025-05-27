package com.example.Backend_SmartVetSystem.service;

import com.example.Backend_SmartVetSystem.dto.request.OwnerRequest;
import com.example.Backend_SmartVetSystem.dto.response.OwnerResponse;
import com.example.Backend_SmartVetSystem.entity.Owner;
import com.example.Backend_SmartVetSystem.exception.AppException;
import com.example.Backend_SmartVetSystem.exception.ErrorCode;
import com.example.Backend_SmartVetSystem.mapper.OwnerMapper;
import com.example.Backend_SmartVetSystem.repository.OwnerRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class OwnerService {
    private final OwnerRepository ownerRepository;
    private final OwnerMapper ownerMapper;
    private final IdGeneratorService idGeneratorService;

    public OwnerResponse createOwner(OwnerRequest request) {
        Owner owner = ownerMapper.toOwner(request);
        owner.setOwnerId(idGeneratorService.generateRandomId("U",ownerRepository::existsById));
        return ownerMapper.toOwnerResponse(ownerRepository.save(owner));
    }

    public List<OwnerResponse> getAllOwners() {
        return ownerRepository.findAll().stream().map(ownerMapper::toOwnerResponse).collect(Collectors.toList());
    }

    public OwnerResponse updateOwner(String ownerId,OwnerRequest request) {
        var owner = ownerRepository.findById(ownerId).orElseThrow(()->new AppException(ErrorCode.OWNER_NOT_FOUND));
        ownerMapper.UpdateOwner(owner,request);
        return ownerMapper.toOwnerResponse(ownerRepository.save(owner));
    }

    public String deleteOwner(String ownerId) {
        ownerRepository.deleteById(ownerId);
        return "owner has Id "+ ownerId + " deleted";
    }

    public OwnerResponse getOwner(String ownerId) {
        Owner owner = ownerRepository.findById(ownerId).orElseThrow(()->new AppException(ErrorCode.OWNER_NOT_FOUND));
        return ownerMapper.toOwnerResponse(owner);
    }
}
