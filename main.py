import cv2
import mediapipe as mp
import math
from pyfirmata2 import Arduino, util
import time

# Arduino setup
board = Arduino('COM14')  # Change the port as needed
servo_pin_left = 2  # Change the pin for the first servo
servo_pin_right = 3  # Change the pin for the second servo
servo_left = board.get_pin('d:{0}:s'.format(servo_pin_left))
servo_right = board.get_pin('d:{0}:s'.format(servo_pin_right))

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# OpenCV setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mapping range for servo (adjust as needed)
servo_min = 0
servo_max = 180

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and get hand landmarks
    results = hands.process(rgb_frame)

    #for face detection
    #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #for (x, y, w, h) in faces:
    #    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the coordinates of index and thumb fingertips
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_finger = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Calculate the average position of thumb and index fingertips
            avg_x = (index_finger.x + thumb_finger.x) / 2
            avg_y = (index_finger.y + thumb_finger.y) / 2

            # Calculate the distance between index and thumb fingertips
            distance = math.sqrt((index_finger.x - thumb_finger.x)**2 + (index_finger.y - thumb_finger.y)**2)

            # Map the distance to the range suitable for servo control (adjust as needed)
            mapped_distance = int(servo_min + (distance / 0.2) * (servo_max - servo_min))
            mapped_distance = max(servo_min, min(mapped_distance, servo_max))

            # Move the servo for the left hand
            if avg_x < 0.5:
                if mapped_distance > 20:
                    servo_left.write(mapped_distance)
                else:
                    servo_left.write(0)
                # Draw a line between index and thumb fingertips for the left hand
                cv2.line(frame, (int(index_finger.x * frame.shape[1]), int(index_finger.y * frame.shape[0])),
                         (int(thumb_finger.x * frame.shape[1]), int(thumb_finger.y * frame.shape[0])),
                         (0, 255, 0), 2)

                # Display the angle of the servo motor for the left hand
                cv2.putText(frame, f'Angulo Izquierdo: {mapped_distance}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Move the servo for the right hand if there is no face detected
            else:
                if mapped_distance > 20:
                    servo_right.write(mapped_distance)
                else:
                    servo_right.write(0)

                # Draw a line between index and thumb fingertips for the right hand
                cv2.line(frame, (int(index_finger.x * frame.shape[1]), int(index_finger.y * frame.shape[0])),
                         (int(thumb_finger.x * frame.shape[1]), int(thumb_finger.y * frame.shape[0])),
                         (255, 0, 0), 2)

                # Display the angle of the servo motor for the right hand
                cv2.putText(frame, f'Angulo Derecho: {mapped_distance}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
board.exit()




