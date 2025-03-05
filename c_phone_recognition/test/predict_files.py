import torch
import torchaudio
from pathlib import Path
import c_phone_recognition.dengine as engine
import c_phone_recognition.utils.config as config
from c_phone_recognition.cmodel import PhoneRecognitionModelResidual
from c_phone_recognition.utils.meltransform import MelTransform
from torchaudio.models.decoder import ctc_decoder
from c_phone_recognition.utils.phonetransform import PhoneTransform


def beam_search_ctc_decoder(emission):
    LEXICON_FILE = "../data/bur_lexicon.txt"
    TOKENS_FILE = "../data/bur_phone_set.txt"

    beam_search_decoder = ctc_decoder(
        lexicon=LEXICON_FILE,
        tokens=TOKENS_FILE,
        lm=None,
        nbest=3,
        beam_size=1500,
        word_score=-0.8,
        blank_token='<BLANK>',
        sil_token='<SIL>',
    )
    emission = emission.permute(1, 0, 2)
    beam_search_result = beam_search_decoder(emission)
    beam_search_tokens = " ".join(beam_search_decoder.idxs_to_tokens(beam_search_result[0][0].tokens))
    beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()
    # print(beam_search_tokens)
    print(beam_search_transcript)


def save_res(path, res):
    with open(path, 'w', encoding='utf-8') as fp:
        fp.write(res.replace('<SPACE>', '|'))


if __name__ == '__main__':
    PHONE_MAP_PATH = '../data/phones_map.json'
    MODEL_PATH = '../models/phone_model_conv_residual.pth'
    AUDIO_DIR = 'C:\\Users\\buryat_saram\\Music\\Buryat Readings\\Testing\\wavs'
    PREDICTED_PHONES_DIR = 'C:\\Users\\buryat_saram\\Music\\Buryat Readings\\Testing\\predicted_phones'
    AUDIO_PATH = '../../00. old/old/old/test_audio/test6.wav'

    audio_files = list(Path(AUDIO_DIR).glob('*.wav'))

    meltransform = MelTransform()
    phonetransform = PhoneTransform(PHONE_MAP_PATH)

    # load and set to eval model
    N_CLASSES = config.NUM_OF_PHONE_UNITS + 1
    model = PhoneRecognitionModelResidual(N_CLASSES)
    model = engine.load_model(model, MODEL_PATH)
    model.eval()

    for fpath in audio_files:
        spectrogram = meltransform.build_spectrogram(audio_path=fpath)
        # specs: (ts, n_channels, n_mels) -> (1, 128, ts)
        spectrogram = spectrogram.permute(1, 2, 0)
        spectrogram = (spectrogram - config.MEAN_VALUE) / config.STD_VALUE
        data = spectrogram.unsqueeze(0)

        log_probs = model(data)


        _, ids = torch.max(log_probs, dim=-1)
        ids = ids.squeeze(1).numpy().tolist()

        res = phonetransform.decode(ids)
        print(res)

        path = Path(PREDICTED_PHONES_DIR) / f'{fpath.stem}.txt'
        save_res(path, res)
        # beam_search_ctc_decoder(log_probs)
