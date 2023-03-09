import streamlit as st
import pyaudio
import wave
import cv2
import numpy as np

# Set parameters for the audio file
channels = 1
sample_width = 2
frame_rate = 44100
chunk = 1024

# Create a new PyAudio object
p = pyaudio.PyAudio()

frames = []

# Define a callback function to handle the audio recording
def audio_callback(in_data, frame_count, time_info, status):
    frames.append(in_data)
    return (in_data, pyaudio.paContinue)

# Define the Streamlit app
def main():
    st.title("Audio Recorder")

    # Create a radio button to start and stop recording
    recording = st.radio("Click to start recording:", ("Start", "Stop"))

    if recording == "Start":
        st.write("Recording...")

        # Create a new wave object and open a new WAV file for writing
        frames = []
        audio = wave.open("recording.wav", "wb")
        audio.setnchannels(channels)
        audio.setsampwidth(sample_width)
        audio.setframerate(frame_rate)

        # Start the audio stream and record audio data
        stream = p.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=frame_rate,
                        input=True,
                        frames_per_buffer=chunk,
                        stream_callback=audio_callback)
        stream.start_stream()

    elif recording == "Stop":
        st.write("Stopped recording.")

        # Stop the audio stream and close the WAV file
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Write the audio frames to the WAV file and close the file
        audio.writeframes(b''.join(frames))
        audio.close()

    st.write("Done.")

if __name__ == "__main__":
    main()