from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

# download and load all models
preload_models()
# voice_preset = "v2/en_speaker_6"

# generate audio from text
text_prompt = """
    [Neutral] Co-fertility is a startup that offers free egg freezing services to women, 
    with a catch: they keep half of the eggs frozen. [Neutral] The startup has bundled together 
    egg freezing and egg donation, allowing women to get free egg freezing if they agree to donate half of 
    their eggs. [Neutral] The donated eggs are then sold to people who want them, with the startup charging 
    $13,700 as a matchmaking fee to connect the donor with the recipient. [Neutral] The startup claims that 
    the best time to freeze eggs is often when women can least afford it, and that it is making egg 
    freezing accessible to all women. [Neutral] While the idea may seem insane, the startup's disruptive
    model has the potential to work as a business, given the growing trend of egg freezing and its high cost. 
    [Concerned] However, some may question the ethics of a startup profiting from the sale of donated eggs and 
    the possible exploitation of women who may feel pressured to donate their eggs for free freezing services.
"""
audio_array = generate_audio(text_prompt)

# save audio to disk
write_wav("bark_gen.wav", SAMPLE_RATE, audio_array)
  
# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)