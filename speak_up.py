import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper
import queue
import os
import wave

# إعدادات التسجيل
sample_rate = 16000  # معدل العينة
silence_threshold = 0.11 # مستوى الصوت الذي يعتبر صمتًا (أقل حساسية)
buffer = queue.Queue()

def memorize(indata, frames, time, status):
    """يتم استدعاؤها عند تسجيل كل دفعة من الصوت"""
    if status:
        print(status)
    buffer.put(indata.copy())

record_audio = []
silence_frames = 0

# بدء التسجيل
with sd.InputStream(samplerate=sample_rate, channels=1, callback=memorize):
    print("Recording... Speak now. The recording will stop when you stop speaking.")

    try:
        while True:
            data = buffer.get()
            record_audio.append(data)

            # حساب أعلى قيمة للصوت في الإطار الحالي
            max_amplitude = np.max(np.abs(data))

            if max_amplitude < silence_threshold:
                silence_frames += len(data)  # زيادة عدد الإطارات الصامتة
            else:
                silence_frames = 0  # إعادة تعيين العداد عند وجود صوت

            # إذا كان هناك صمت تام لفترة طويلة، توقف
            if silence_frames > 10 * sample_rate:  # 10 ثوانٍ من الصمت المستمر
                print("Silence detected, stopping recording...")
                break

    except KeyboardInterrupt:
        print("Recording stopped manually.")

# دمج البيانات الصوتية وتحويلها إلى int16
recorded_audio = np.concatenate(record_audio, axis=0)
recorded_audio = (recorded_audio * 32767).astype(np.int16)

# حفظ الملف الصوتي
audio_path = os.path.abspath("recorded_audio.wav")
wav.write(audio_path, sample_rate, recorded_audio)

# التأكد من أن الملف تم إنشاؤه بنجاح
if os.path.exists(audio_path):
    print(f"Audio file saved successfully at: {audio_path}")
else:
    print("Failed to save audio file.")
    exit(1)

# التحقق مما إذا كان الملف الصوتي تالفًا
try:
    with wave.open(audio_path, "rb") as wf:
        print(f"Audio file is valid, duration: {wf.getnframes() / wf.getframerate()} seconds")
except wave.Error as e:
    print("Error: Corrupted WAV file:", e)
    exit(1)

# تحميل موديل Whisper
print("Loading Whisper model...")
model = whisper.load_model("small")

# التأكد من أن ffmpeg مثبت ويمكن الوصول إليه
ffmpeg_check = os.system("ffmpeg -version")
if ffmpeg_check != 0:
    print("Error: ffmpeg is not installed or not found in PATH.")
    exit(1)

# محاولة تحويل الصوت إلى نص
try:
    result = model.transcribe(audio_path)
    print("Transcription: ", result['text'])
except Exception as e:
    print("Error during transcription:", e)