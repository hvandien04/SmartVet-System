package com.example.Backend_SmartVetSystem;

import io.github.cdimascio.dotenv.Dotenv;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class BackendSmartVetSystemApplicationTests {

	public static void main(String[] args) {

		Dotenv dotenv = Dotenv.configure()
				.filename(".env") // Tên file .env của bạn
				.load();

		System.setProperty("URL_DB", dotenv.get("URL_DB"));
		System.setProperty("USERNAME_DB", dotenv.get("USERNAME_DB"));
		System.setProperty("PASSWORD_DB", dotenv.get("PASSWORD_DB"));
		System.setProperty("SENDGRID_API_KEY", dotenv.get("SENDGRID_API_KEY"));
		System.setProperty("FROM_EMAIL", dotenv.get("FROM_EMAIL"));
		System.setProperty("SIGNER_KEY", dotenv.get("SIGNER_KEY"));
		System.setProperty("DOMAIN_FE", dotenv.get("DOMAIN_FE"));
		SpringApplication.run(BackendSmartVetSystemApplicationTests.class, args);
	}

}
