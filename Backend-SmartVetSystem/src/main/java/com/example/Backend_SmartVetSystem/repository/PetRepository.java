package com.example.Backend_SmartVetSystem.repository;

import com.example.Backend_SmartVetSystem.entity.Pet;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface PetRepository extends JpaRepository<Pet, String> {
}
