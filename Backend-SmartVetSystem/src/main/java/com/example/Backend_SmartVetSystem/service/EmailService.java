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
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;

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


    public void sendAppointmentReminder(String toEmail, String appointmentTime) throws IOException {
        if (fromEmail == null || fromEmail.isEmpty()) {
            throw new IOException("FROM_EMAIL not configured yet.");
        }

        Instant time = Instant.parse(appointmentTime);
        String formatted = DateTimeFormatter.ofPattern("HH:mm 'ngày' dd/MM/yyyy")
                .withZone(ZoneId.of("UTC"))
                .format(time);

        String subject = "Nhắc lịch hẹn khám thú y tại SmartVet";

        String htmlContent = """
        <div style="font-family: Arial, sans-serif; font-size: 14px; color: #333;">
            <p>Kính gửi Quý khách,</p>
            <p>SmartVet xin gửi lời chào trân trọng đến Quý khách!</p>
            <p>Hệ thống ghi nhận Quý khách đã đặt lịch hẹn khám bệnh thú y cho thú cưng tại SmartVet. Đây là email tự động nhằm nhắc lịch hẹn khám như sau:</p>
            
            <p>🕒 <strong>Thời gian khám dự kiến:</strong> %s<br>
            🏥 <strong>Địa điểm:</strong> SmartVet – 123 Tô Ký, Phường Tân Chánh Hiệp, Quận 12, TP.HCM</p>
            
            <p>Để buổi khám diễn ra thuận lợi, Quý khách vui lòng đến đúng giờ hoặc đến sớm 10 phút để làm thủ tục. Trong trường hợp cần hỗ trợ hoặc thay đổi lịch hẹn, xin liên hệ:</p>
            
            <p>📞 Hotline: 0909 123 456<br>
            📧 Email: support@smartvet.vn</p>

            <p>Cảm ơn Quý khách đã tin tưởng và sử dụng dịch vụ của SmartVet!<br>
            Chúc thú cưng của bạn luôn khỏe mạnh và vui vẻ.</p>
            
            <p>Trân trọng,<br>
            <strong>Đội ngũ SmartVet</strong></p>
        </div>
    """.formatted(formatted);

        Email from = new Email(fromEmail);
        Email to = new Email(toEmail);
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
