import serial
import time

class ServoController:
    def __init__(self, port='COM4', baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Wait for Arduino
        # --- ADD THIS LINE TO CLEAR THE BUFFER ---
        self.ser.flushInput() # Clears any incoming data in the buffer
        print(f"Connected to {port} at {baudrate} baud")

    def send_command(self, command):
        self.ser.write(command.encode())  # Send raw command (no newline)
        time.sleep(0.1)  # Short delay
        response = self.ser.readline().decode().strip()
        if response:
            print(f"Arduino: {response}")

    def close(self):
        self.ser.close()
        print("Connection closed")

if __name__ == "__main__":
    controller = ServoController(port='COM4')  # Change to your port

    try:
        while True:
            print("\n--- Servo Control Menu ---")
            print("1: Toggle Servo 1 (90°/180°)")
            print("2: Toggle Servo 2 (90°/180°)")
            print("3: Toggle Servo 3 (90°/180°)")
            print("4: Toggle Servo 4 (90°/180°)")
            print("5: Toggle Servo 5 (90°/180°)")
            print("0: Center all servos (90°)")
            print("s: Toggle spin mode")
            print("q: Quit")

            choice = input("Enter command (1/2/3/4/5/0/s/q): ").strip()  
            if not choice:
                continue  
            
            if choice.lower() == 'q':
                break
            elif choice in ['0', '1', '2', '3', '4', '5', 's']:
                controller.send_command(choice)  
            else:
                print("Invalid command!")

    finally:
        controller.close()