#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// PCA9685 Settings
#define PCA9685_ADDRESS 0x40
#define PCA9685_FREQUENCY 50

// Servo Pulse Calibration (adjust these if your servo's 0° and 180° positions are off)
#define SERVO_MIN_PULSE 150
#define SERVO_MAX_PULSE 600

// Servo Channel Assignments (match these to how you wired your servos to the PCA9685)
#define SERVO1_CHANNEL 0
#define SERVO2_CHANNEL 1
#define SERVO3_CHANNEL 2
#define SERVO4_CHANNEL 3
#define SERVO5_CHANNEL 4

#define TRIGGER_SPEED_MS 6   

#define RETURN_SPEED_MS 10   // Example: Slower movement back to default (higher value = slower)


Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA9685_ADDRESS);

uint8_t servo1_default_angle = 85;  
uint8_t servo2_default_angle = 120; 
uint8_t servo3_default_angle = 32;  
uint8_t servo4_default_angle = 25;
uint8_t servo5_default_angle = 22;
uint8_t servo1_current_angle = servo1_default_angle;
uint8_t servo2_current_angle = servo2_default_angle;
uint8_t servo3_current_angle = servo3_default_angle;
uint8_t servo4_current_angle = servo4_default_angle;
uint8_t servo5_current_angle = servo5_default_angle;

// Function prototypes
void moveServo(uint8_t servoNum, uint8_t angle);
// moveServoSlowly now takes an additional parameter for the specific delay to use
void moveServoSlowly(uint8_t servoNum, uint8_t startAngle, uint8_t targetAngle, uint16_t stepDelayMs);
void spinAllServos();

bool spinning = false; // Flag for continuous spin mode

void setup() {
  Serial.begin(115200); // Initialize serial for communication (e.g., with Python)
  Wire.begin();
  pwm.begin();
  pwm.setPWMFreq(PCA9685_FREQUENCY);
  
  // Set all servos to their defined default positions on Arduino startup
  moveServo(SERVO1_CHANNEL, servo1_default_angle);
  moveServo(SERVO2_CHANNEL, servo2_default_angle);
  moveServo(SERVO3_CHANNEL, servo3_default_angle);
  moveServo(SERVO4_CHANNEL, servo4_default_angle);
  moveServo(SERVO5_CHANNEL, servo5_default_angle);
}

void loop() {
  if (Serial.available() > 0) {
    char input = Serial.read();
    
    // Stop spinning mode if a direct servo command is received
    if (input == '1' || input == '2' || input == '3' || input == '4' || input == '5' || input == '0') {
      spinning = false;
    }

    switch(input) {
      case '1':
        // Move to test position using TRIGGER_SPEED_MS
        moveServoSlowly(SERVO1_CHANNEL, servo1_current_angle, 20, 0);
        servo1_current_angle = 20;
        delay(2500); // Hold for 3 seconds
        // Return to default position using RETURN_SPEED_MS
        moveServoSlowly(SERVO1_CHANNEL, servo1_current_angle, servo1_default_angle, RETURN_SPEED_MS);
        servo1_current_angle = servo1_default_angle;
        break;
      case '2':
        // Move to test position using TRIGGER_SPEED_MS
        moveServoSlowly(SERVO2_CHANNEL, servo2_current_angle, 60, 10);
        servo2_current_angle = 60;
        delay(5000); // Hold for 3 seconds
        // Return to default position using RETURN_SPEED_MS
        moveServoSlowly(SERVO2_CHANNEL, servo2_current_angle, servo2_default_angle, 0);
        servo2_current_angle = servo2_default_angle;
        break;
      case '3':
        // Move to test position using TRIGGER_SPEED_MS
        moveServoSlowly(SERVO3_CHANNEL, servo3_current_angle, 100, 0);
        servo3_current_angle = 100;
        delay(6000); // Hold for 6 seconds
        // Return to default position using RETURN_SPEED_MS
        moveServoSlowly(SERVO3_CHANNEL, servo3_current_angle, servo3_default_angle, 0);
        servo3_current_angle = servo3_default_angle;
        break;
      case '4':
        // Move to test position using TRIGGER_SPEED_MS
        moveServoSlowly(SERVO4_CHANNEL, servo4_current_angle, 100, TRIGGER_SPEED_MS);
        servo4_current_angle = 100;
        delay(4500); // Hold for 3 seconds
        // Return to default position using RETURN_SPEED_MS
        moveServoSlowly(SERVO4_CHANNEL, servo4_current_angle, servo4_default_angle, 8);
        servo4_current_angle = servo4_default_angle;
        break;
      case '5':
        // Move to test position using TRIGGER_SPEED_MS
        moveServoSlowly(SERVO5_CHANNEL, servo5_current_angle, 100, TRIGGER_SPEED_MS);
        servo5_current_angle = 100;
        delay(2000); // Hold for 3 seconds
        // Return to default position using RETURN_SPEED_MS
        moveServoSlowly(SERVO5_CHANNEL, servo5_current_angle, servo5_default_angle, 5);
        servo5_current_angle = servo5_default_angle;
        break;
      case '0':
        {
          // Move all servos to their individual default positions slowly, using RETURN_SPEED_MS
          uint8_t* currentAngles[] = {
            &servo1_current_angle, &servo2_current_angle, &servo3_current_angle,
            &servo4_current_angle, &servo5_current_angle
          };
          uint8_t defaultAngles[] = {
            servo1_default_angle, servo2_default_angle, servo3_default_angle,
            servo4_default_angle, servo5_default_angle
          };

          int max_distance = 0;
          for (int i = 0; i < 5; i++) {
            int distance = abs(*currentAngles[i] - defaultAngles[i]);
            if (distance > max_distance) {
              max_distance = distance;
            }
          }

          for (int step_count = 0; step_count <= max_distance; step_count++) {
            for (int i = 0; i < 5; i++) {
              int current_servo_angle = *currentAngles[i];
              uint8_t targetAngle = defaultAngles[i];
              
              if (current_servo_angle != targetAngle) {
                int step_direction = (targetAngle > current_servo_angle) ? 1 : -1;
                int next_angle = current_servo_angle + step_direction;
                
                if (step_direction == 1 && next_angle > targetAngle) next_angle = targetAngle;
                if (step_direction == -1 && next_angle < targetAngle) next_angle = targetAngle;

                moveServo(i, next_angle);
                *currentAngles[i] = next_angle;
              }
            }
            delay(RETURN_SPEED_MS); // Use RETURN_SPEED_MS for the overall centering movement
          }

          // Ensure current state variables are precisely updated to default values
          servo1_current_angle = servo1_default_angle;
          servo2_current_angle = servo2_default_angle;
          servo3_current_angle = servo3_default_angle;
          servo4_current_angle = servo4_default_angle;
          servo5_current_angle = servo5_default_angle;
        }
        break;
      case 's':
        spinning = !spinning; // Toggle spin mode on/off
        break;
      default:
        break; // No output for invalid input when controlled by Python
    }
  }

  // If spin mode is active, run the continuous spin function
  if (spinning) {
    spinAllServos();
  }
}

// Moves a servo instantly to the specified angle.
void moveServo(uint8_t servoNum, uint8_t angle) {
  angle = constrain(angle, 0, 180);
  uint16_t pulse = map(angle, 0, 180, SERVO_MIN_PULSE, SERVO_MAX_PULSE);
  pwm.setPWM(servoNum, 0, pulse);
}

// Moves a servo gradually from its current angle to a target angle using a specified step delay.
// stepDelayMs: The delay in milliseconds to pause between each degree step.
void moveServoSlowly(uint8_t servoNum, uint8_t startAngle, uint8_t targetAngle, uint16_t stepDelayMs) {
  if (startAngle == targetAngle) return;

  int step_direction = (targetAngle > startAngle) ? 1 : -1;
  
  for (int angle = startAngle; angle != targetAngle + step_direction; angle += step_direction) {
    moveServo(servoNum, angle);
    delay(stepDelayMs); // Use the provided stepDelayMs for this specific movement
  }
  moveServo(servoNum, targetAngle); // Ensure final position is exact
}

// Makes all servos sweep back and forth continuously, using RETURN_SPEED_MS.
void spinAllServos() {
  static uint8_t angle = 0;
  static bool increasing = true;

  for (int i = 0; i < 5; i++) {
    moveServo(i, angle);
  }

  if (increasing) {
    angle++;
    if (angle >= 180) increasing = false;
  } else {
    angle--;
    if (angle <= 0) increasing = true;
  }

  delay(RETURN_SPEED_MS); // Continuous spin uses the RETURN_SPEED_MS
}