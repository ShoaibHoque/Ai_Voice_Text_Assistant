# updated_voice_processing.py
import whisper
from gtts import gTTS
import sounddevice as sd
import numpy as np
import os

class VoiceProcessor:
    def __init__(self, model_name="medium"):
        # Load Whisper model
        self.whisper_model = whisper.load_model(model_name)

    def record_audio(self, duration=5, samplerate=16000):
        print("Recording audio...")
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
        sd.wait()
        print("Recording complete.")
        return audio.flatten()

    def transcribe_audio(self, audio, samplerate=16000):
        # Save audio to a temporary WAV file
        temp_audio_path = "temp_audio.wav"
        audio_data = (audio * 32767).astype(np.int16)  # Convert to 16-bit PCM
        sd.write(temp_audio_path, audio_data, samplerate)

        # Transcribe using Whisper
        result = self.whisper_model.transcribe(temp_audio_path, language="en")  # Change to "ar" for Arabic
        os.remove(temp_audio_path)  # Cleanup temp file
        return result["text"]

    def synthesize_speech(self, text, lang="en", output_path="response.mp3"):
        # Use gTTS to convert text to speech
        tts = gTTS(text=text, lang=lang)
        tts.save(output_path)
        print(f"Response audio saved to {output_path}.")
        return output_path