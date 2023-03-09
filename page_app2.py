import speech_recognition as sr
import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the speech recognizer
r = sr.Recognizer()

# Define the voice commands
start_cmd = "start recording"
stop_cmd = "stop recording"

while True:
    # Capture the video frames
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Display the frame
    cv2.imshow('frame', gray)
    
    # Use the speech recognizer to listen for voice commands
    with sr.Microphone() as source:
        audio = r.listen(source)
    
    try:
        # Convert speech to text
        command = r.recognize_google(audio)
        print(command)
        
        # Check for voice commands
        if command == start_cmd:
            # Start recording video
            out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (640,480))
            break
        elif command == stop_cmd:
            # Stop recording video and save it
            out.release()
            break
    except:
        pass
    
    # Wait for key press to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()



