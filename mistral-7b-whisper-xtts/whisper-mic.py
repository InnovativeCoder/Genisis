from whisper_mic import WhisperMic
import requests
import json

mic = WhisperMic()
result = mic.listen()
print(result)

transcription_text = result

prompt = "Assume you are a very polite waitress who is sarcastic, naughty and jolly and answers briefly, if you want to laugh instead replace it with [laugh], now answer : " + transcription_text

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