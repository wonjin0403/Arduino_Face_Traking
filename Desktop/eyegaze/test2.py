import cv2
import serial
import numpy as np

# Update the port to match your specific port
ARDUINO_PORT = '/dev/cu.usbserial-14110'

# Open serial connection to Arduino
arduino = serial.Serial(ARDUINO_PORT, 115200, timeout=1)

def read_camera_frame():
    # Send a command to the Arduino to capture a frame
    arduino.write(b'C\n')

    # Read the image size (assuming a fixed size for simplicity)
    image_size = int(arduino.readline().decode().strip())

    # Read the image data
    image_data = arduino.read(image_size)

    # Convert the image data to a NumPy array
    image_array = np.frombuffer(image_data, dtype=np.uint8)

    # Decode the image array
    frame = cv2.imdecode(image_array, 1)

    return frame

def main():
    while True:
        # Read a frame from the Arduino
        frame = read_camera_frame()

        # Display the frame
        print(frame.shape)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the serial connection and close the window
    arduino.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()