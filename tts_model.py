# Import necessary modules
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.audio import AudioProcessor
import TTS

# Path to the model files (use Coqui's pre-trained model paths here)
model_name = "tts_models/en/ljspeech/tacotron2-DCA"  # Feel free to use another model
model_path = TTS.utils.download_model(model_name)
config_path = model_path.replace(".pth.tar", ".json")

# Initialize the synthesizer
synthesizer = Synthesizer(model_path, config_path)

# Text to synthesize
text = "Hello, this is a test of our TTS setup."

# Generate audio
wav = synthesizer.tts(text)

# Save audio to a file
synthesizer.save_wav(wav, "output_test.wav")

print("Synthesis complete. Check 'output_test.wav' for results.")
