from vad import VoiceActivityDetector

filename = './448_000046.wav'
v = VoiceActivityDetector(filename)
# v.plot_detected_speech_regions()
print(v.convert_windows_to_readible_labels(v.detect_speech()))

from pydub import AudioSegment,silence

myaudio = AudioSegment.from_wav("448_000046.wav")

silence = silence.detect_silence(myaudio, min_silence_len=10, silence_thresh=-50)

silence = [((start/10),(stop/10)) for start,stop in silence] #convert to sec
print(silence)