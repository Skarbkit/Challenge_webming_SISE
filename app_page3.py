import streamlit as st
import speech_recognition as sr
import cv2
import threading

# Define the voice commands
start_cmd = "start recording"
stop_cmd = "stop recording"

def record_video():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Start recording video
    out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (640,480))
    
    while True:
        # Capture the video frames
        ret, frame = cap.read()
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Display the frame
        cv2.imshow('frame', gray)
        
        # Check for key press to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Stop recording video and release the webcam
            out.release()
            cap.release()
            cv2.destroyAllWindows()
            break
        
        # Wait for key press to close the window
        if cv2.waitKey(1) & 0xFF == ord('c'):
            # Close the window and release the webcam
            cap.release()
            cv2.destroyAllWindows()
            break
        
        # Record the frame
        out.write(frame)

def voice_recognition():
    # Initialize the speech recognizer
    r = sr.Recognizer()
    
    while True:
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
                t = threading.Thread(target=record_video)
                t.start()
            elif command == stop_cmd:
                # Stop recording video and save it
                cv2.imwrite('last_frame.jpg', cv2.imread('output.avi'))
                break
        except:
            pass

# Define the Streamlit app
def main():
    st.title("Webcam Recorder")
    
    # Start voice recognition
    st.write("Say 'start recording' to start recording and 'stop recording' to stop and save the video.")
    t = threading.Thread(target=voice_recognition)
    t.start()
    
    # Show the last recorded frame
    if st.button("Show last recorded frame"):
        st.image('last_frame.jpg')
    
if __name__ == '__main__':
    main()
