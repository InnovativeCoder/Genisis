from whisper_mic import WhisperMic
import requests
import json
import requests
import torch
from TTS.api import TTS

#todo - pyAudio get the mic source and destination

# Get device
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # List available 🐸TTS models
# print(TTS().list_models())

# # Init TTS
# tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
# tts.tts(text="Hello world!", speaker_wav="./bark_out.wav", language="en")

mic = WhisperMic()
result = mic.listen()
print(result)

transcription_text = result


llmmodel = 'mistral'
try:
    r = requests.post('http://localhost:11434/api/generate',
                      json={
                          'model': llmmodel,
                          'prompt': prompt,
                      },
                      stream=True)
    r.raise_for_status()
    response_text = str(r.text)

    for line in r.iter_lines():
        body = json.loads(line)
        response_part = body.get('response', '')
        # Process and print each token from the streamed response
        print(response_part, end='', flush=True)

except requests.exceptions.RequestException as e:
    print(f"Error during API request: {e}")