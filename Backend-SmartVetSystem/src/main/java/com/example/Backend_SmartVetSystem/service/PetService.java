package com.example.Backend_SmartVetSystem.service;

import com.example.Backend_SmartVetSystem.dto.request.PetCreateRequest;
import com.example.Backend_SmartVetSystem.dto.request.PetUpdateRequest;
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

import java.util.List;
import java.util.stream.Collectors;

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


    public List<PetResponse> GetAllPets() {
        return petRepository.findAll().stream().map(petMapper::toPetResponse).collect(Collectors.toList());
    }

    public PetResponse getPet(String petId) {
        Pet pet = petRepository.findById(petId).orElseThrow(()-> new AppException(ErrorCode.PET_NOT_FOUND));
        return petMapper.toPetResponse(pet);
    }

    public PetResponse updatePet(String petId, PetUpdateRequest request) {
        Pet pet = petRepository.findById(petId).orElseThrow(()->new AppException(ErrorCode.PET_NOT_FOUND));
        petMapper.UpdatePet(pet, request);
        if(request.getOwnerId() != null) {
            Owner owner = ownerRepository.findById(request.getOwnerId()).orElseThrow(()->new AppException(ErrorCode.OWNER_NOT_FOUND));
            pet.setOwner(owner);
        }
        return petMapper.toPetResponse(petRepository.save(pet));
    }

    public String deletePet(String petId) {
        petRepository.deleteById(petId);
        return "Pet has Id " + petId + " deleted";
    }

    public List<PetResponse> getPetsByOwner(String ownerId) {
        Owner owner = ownerRepository.findById(ownerId).orElseThrow(()->new AppException(ErrorCode.OWNER_NOT_FOUND));
        return petRepository.findAllByOwner(owner).stream().map(petMapper::toPetResponse).collect(Collectors.toList());
    }
}
