from pathlib import Path
import re

path = "C:\\Users\\buryat_saram\\Music\\Project Buryat Saram\\buryat_text_to_speech\\temp_wavs"


audio_files = list(Path(path).glob('*.wav'))
print(len(audio_files))


for audio in audio_files:
    stem = audio.stem
    if len(stem.split('.')) > 1:
        name, id = stem.split('.')
        id = re.sub('^0*', '', id)
        new_file_path = Path(path) / f'{name}_{id}.wav'

        print(new_file_path)
        audio.rename(new_file_path)
