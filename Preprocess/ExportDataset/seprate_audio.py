import os
from moviepy.editor import AudioFileClip



def separate_audio(origin_path, target_path, *args, **kwargs):
    audio_clip = AudioFileClip(origin_path)
    audio_clip.write_audiofile(target_path)
