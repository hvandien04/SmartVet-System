package com.example.Backend_SmartVetSystem.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDate;

@Getter
@Setter
@Entity
@AllArgsConstructor
@NoArgsConstructor
@Builder
@Table(name = "pet")
public class Pet {
    @Id
    @Column(name = "pet_id", nullable = false, length = 50)
    private String petId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "owner_id")
    private Owner owner;

    @Column(name = "name", nullable = false)
    private String name;

    @Column(name = "species", length = 50)
    private String species;

    @Column(name = "breed", length = 50)
    private String breed;

    @Column(name = "gender", length = 10)
    private String gender;

    @Column(name = "birth_date")
    private LocalDate birthDate;

}