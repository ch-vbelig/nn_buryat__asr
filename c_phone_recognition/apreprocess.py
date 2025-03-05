import numpy as np
from bdataset import SpeechDataset

def calculate_norm_values(data):
    x = []
    for d in data:
        x.extend(d)
    data = np.array(x)

    # Calculate the mean and std values
    mean = data.mean()
    std = data.std()

    return mean, std


def run_preprocess():
    AUDIO_DIR = 'data/audio'
    PHONES_DIR = 'data/phones'

    dataset = SpeechDataset(
        audio_dir=AUDIO_DIR,
        phone_dir=PHONES_DIR,
    )

    spectrograms = []

    for i in range(len(dataset)):
        spec, _, _, _= dataset[i]
        spectrograms.append(spec)

    mean, std = calculate_norm_values(spectrograms)
    print('Mean:', mean)
    print('Std:', std)


if __name__ == '__main__':
    run_preprocess()

