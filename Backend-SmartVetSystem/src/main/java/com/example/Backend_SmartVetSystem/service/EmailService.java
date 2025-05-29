package com.example.Backend_SmartVetSystem.service;

import com.sendgrid.*;
import com.sendgrid.helpers.mail.Mail;
import com.sendgrid.helpers.mail.objects.Content;
import com.sendgrid.helpers.mail.objects.Email;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import lombok.RequiredArgsConstructor;
import lombok.experimental.NonFinal;

import java.io.IOException;

@Service
@RequiredArgsConstructor
public class EmailService {

    @NonFinal
    @Value("${spring.sendgrid.api-key}")
    private String sendgridApiKey;

    @NonFinal
    @Value("${FROM_EMAIL}")
    private String fromEmail;

    public void sendResetCode(String toEmail, String code) throws IOException {
        if (fromEmail == null || fromEmail.isEmpty()) {
            throw new IOException("FROM_EMAIL not configured yet.");
        }

        Email from = new Email(fromEmail);
        String subject = "Password reset confirmation code";
        Email to = new Email(toEmail);
        String htmlContent = """
            <div style="font-family: Arial, sans-serif; font-size: 14px;">
                <p>Your code will expire in <strong>5 minutes</strong>.</p>
                <p>Your verification code is:</p>
                <p style="font-size: 24px; font-weight: bold; color: #2d3748;">%s</p>
            </div>
        """.formatted(code);

        Content content = new Content("text/html", htmlContent);

        Mail mail = new Mail(from, subject, to, content);

        SendGrid sg = new SendGrid(this.sendgridApiKey);
        Request request = new Request();

        request.setMethod(Method.POST);
        request.setEndpoint("mail/send");
        request.setBody(mail.build());
        sg.api(request);
    }


}
