from pathlib import Path

def get_files(dir):
    audio_files = list(Path(dir).glob('*wav'))
    return audio_files

def rename(audio_files, speaker_id):
    for i, fname in enumerate(audio_files):
        file_id = str(i+1).rjust(4, '0')
        new_path = Path(fname.parent) / f'id_{speaker_id}_{file_id}.wav'
        fname.rename(new_path)

if __name__ == '__main__':
    DIR_PATH = './audio2/'
    audio_files = get_files(DIR_PATH)

    rename(audio_files, 1)