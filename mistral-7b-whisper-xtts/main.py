import whisper
import requests

modelname = "tiny"
model = whisper.load_model(modelname)
result = model.transcribe(filename, fp16=False, language="en")
transcription_text = result['text']

prompt = "translate this English text into French: " + transcription_text

llmmodel = 'llama2'
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

