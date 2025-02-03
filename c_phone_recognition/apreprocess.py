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
    MAP_SAVE_PATH = 'data/saved_maps.json'

    dataset = SpeechDataset(
        audio_dir=AUDIO_DIR,
        phone_dir=PHONES_DIR,
        phone_map_path=MAP_SAVE_PATH
    )
    spectrograms = []

    for i in range(len(dataset)):
        spec, _, _, _= dataset[i]
        spectrograms.append(spec)

    n_classes = dataset.n_classes
    print('N_classes:', n_classes)
    # dataset.save_maps()

    mean, std = calculate_norm_values(spectrograms)
    print('Mean:', mean)
    print('Std:', std)


if __name__ == '__main__':
    run_preprocess()

