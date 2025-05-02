import numpy as np
import matplotlib.pyplot as plt
import re

# Parameters
bit_duration = 1 / 44000  # duration of each bit in seconds
sample_rate = 44100  # samples per second
high_amp = 0.5
low_amp = -0.5

# Load and extract bit content from text file
with open('./log-2025-05-03T01:02:58.txt', 'r') as f:
    content = f.read()

snippets = re.findall(r"<\|sot\|>(.*?)<\|eot\|>", content, re.DOTALL)
text_to_analyze = ''.join(snippets)
bitstream = ''.join(format(ord(c), '08b') for c in text_to_analyze)

# Convert bitstream into signal
samples = []
for bit in bitstream:
    amp = high_amp if bit == '1' else low_amp
    samples.extend([amp] * int(sample_rate * bit_duration))
samples = np.array(samples, dtype=np.float32)

# Perform spectrogram
plt.figure(figsize=(12, 5))
plt.specgram(samples, NFFT=1024, Fs=sample_rate, noverlap=512, cmap='viridis')
plt.title("Spectral Analysis of Bitstream from <|sot|> Blocks")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label='Intensity (dB)')
plt.tight_layout()
plt.show()
