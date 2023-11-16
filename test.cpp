#include <Adafruit_PWMServoDriver.h>
#include <Servo.h>
#include <Wire.h>
#include "esp_camera.h"
#include <Arduino.h>

Adafruit_PWMServoDriver pwm= Adafruit_PWMServoDriver();
int servoMIN = 150;
int servoMAX = 600;
int onecy = 180;
int ang = 0;

int zeroset = constrain(map(ang,0,180,150,600),150,600);
int onecyset = constrain(map(onecy,0,180,150,600),150,600);
int xzeroset = constrain(map(60,0,180,150,600),150,600);
int xonecyset = constrain(map(120,0,180,150,600),150,600);

// Function to send image data over serial
void sendImage() {
    // Capture an image
    Serial.println("Capturing image...");
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Failed to capture image");
        delay(1000);
        return;
    }

    // Send the image size over serial
    Serial.write((uint8_t *)&fb->len, sizeof(size_t));

    // Send the image data over serial
    Serial.write(fb->buf, fb->len);

    // Release the camera buffer
    esp_camera_fb_return(fb);
}

void move_motor(xy_angle, z_angle){
    #angl(4,50);
    #angl(5,50);
    angl(4,00);
    angl(5,00);
    angl(0,z_angle+90);
    angl(1,z_angle+85);
    angl(2,xy_angle+90);
    angl(3,xy_angle+80);
}

int anc(int i){
  int ac = i;
  return constrain(map(ac,0,180,150,600),150,600);
}
void angl(int a,int b){
  int c = anc(b);
  int value = c;
  pwm.setPWM(a,0,value);
}

void processCommand(char command) {
    switch (command) {
        case 'C':
            // Send image when 'C' is received
            sendImage();
            break;
        // Add more cases for other commands if needed

        case 'P':
            byte x_angle = Serial.read();
            byte y_angle = Serial.read();
            // Convert the bytes back to integers
            int xy_normalized = int(x_byte);
            int z_normalized = int(y_byte);
            move_motor(xy_normalized, z_normalized)

        default:
            // Handle unknown command
            Serial.println("Unknown command");
    }
}

void setup() {
    Serial.begin(115200); // Set the baud rate to match the Python code
    pwm.begin ();
    pwm.setPWMFreq(60);

}

void loop() {
    if (Serial.available() > 0) {
        char command = Serial.read();
        // Print the received command for debugging
        Serial.print("Received command: ");
        Serial.println(command);

        // Process the received command
        processCommand(command);

        // Wait for a while before capturing the next image
        delay(5000);

    }

}

