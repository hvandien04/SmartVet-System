package com.example.Backend_SmartVetSystem.repository;

import com.example.Backend_SmartVetSystem.entity.Owner;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface OwnerRepository extends JpaRepository<Owner, String> {
}
