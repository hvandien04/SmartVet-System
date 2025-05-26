package com.example.Backend_SmartVetSystem.service;

import com.example.Backend_SmartVetSystem.dto.request.PetCreateRequest;
import com.example.Backend_SmartVetSystem.dto.response.PetResponse;
import com.example.Backend_SmartVetSystem.entity.Owner;
import com.example.Backend_SmartVetSystem.entity.Pet;
import com.example.Backend_SmartVetSystem.exception.AppException;
import com.example.Backend_SmartVetSystem.exception.ErrorCode;
import com.example.Backend_SmartVetSystem.mapper.OwnerMapper;
import com.example.Backend_SmartVetSystem.mapper.PetMapper;
import com.example.Backend_SmartVetSystem.repository.OwnerRepository;
import com.example.Backend_SmartVetSystem.repository.PetRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class PetService {
    public final PetRepository petRepository;
    public final PetMapper petMapper;
    public final OwnerRepository ownerRepository;
    public final OwnerMapper ownerMapper;
    public final IdGeneratorService idGeneratorService;

    public PetResponse createPet(PetCreateRequest request) {
        Pet pet = petMapper.toPet(request);
        pet.setPetId(idGeneratorService.generateRandomId("P",petRepository::existsById));

        if (request.getOwnerId() == null) {
            Owner owner = ownerMapper.toOwner(request.getOwner());
            owner.setOwnerId(idGeneratorService.generateRandomId("O",ownerRepository::existsById));
            pet.setOwner(owner);
            ownerRepository.save(owner);
        } else {
            var owner = ownerRepository.findById(request.getOwnerId()).orElseThrow(()->new AppException(ErrorCode.OWNER_NOT_FOUND));
            pet.setOwner(owner);
        }
        petRepository.save(pet);
        return petMapper.toPetResponse(pet);
    }
}
