import pyaudio
import numpy as np
import re

# Parameters
bit_duration = 1/44000  # seconds per bit
sample_rate = 44100  # Hz
high_amp = 0.5
low_amp = -0.5

# Open the log file and extract only <|sot|> ... <|eot|> sections
with open('./research_records/log-2025-05-01T00:04:41.txt', 'r') as f:
    content = f.read()

# Use regex to extract all SOT-EOT blocks
snippets = re.findall(r"<\|sot\|>(.*?)<\|eot\|>", content, re.DOTALL)

# Combine all snippets into one string
playback_text = ''.join(snippets)

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=sample_rate,
                output=True)

def bit_to_wave(bit):
    amp = high_amp if bit == '1' else low_amp
    samples = np.full(int(sample_rate * bit_duration), amp, dtype=np.float32)
    return samples

# Convert each character in the extracted segments to binary and play
for char in playback_text:
    binary = format(ord(char), '08b')
    for bit in binary:
        wave_data = bit_to_wave(bit)
        stream.write(wave_data.tobytes())

stream.stop_stream()
stream.close()
p.terminate()
