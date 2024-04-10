from transformers import AutoProcessor, BarkModel

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

voice_preset = "v2/en_speaker_6"

# inputs = processor("[clears throat] Hello, my dog is cute", voice_preset=voice_preset)
inputs = processor("[Neutral] Co-fertility is a startup that offers free egg freezing services to women", voice_preset=voice_preset)
# inputs = processor("[Neutral] Co-fertility is a startup that offers free egg freezing services to women, with a catch: they keep half of the eggs frozen. [Neutral] The startup has bundled together egg freezing and egg donation, allowing women to get free egg freezing if they agree to donate half of their eggs. [Neutral] The donated eggs are then sold to people who want them, with the startup charging $13,700 as a matchmaking fee to connect the donor with the recipient. [Neutral] The startup claims that the best time to freeze eggs is often when women can least afford it, and that it is making egg freezing accessible to all women. [Neutral] While the idea may seem insane, the startup's disruptive model has the potential to work as a business, given the growing trend of egg freezing and its high cost. [Concerned] However, some may question the ethics of a startup profiting from the sale of donated eggs and the possible exploitation of women who may feel pressured to donate their eggs for free freezing services.", voice_preset=voice_preset)

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

import scipy

sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("yt_bark.wav", rate=sample_rate, data=audio_array)