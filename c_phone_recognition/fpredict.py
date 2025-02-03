import torch
import torchaudio
from pathlib import Path
import dengine as engine
import utils.config as config
from cmodel import PhoneRecognitionModelResidual
from utils.meltransform import MelTransform
from torchaudio.models.decoder import ctc_decoder
from utils.phonetransform import PhoneTransform


def beam_search_ctc_decoder(emission):
    LEXICON_FILE = "decoder/bur_lexicon_2.txt"
    TOKENS_FILE = "decoder/bur_phone_set.txt"

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
    print(beam_search_tokens)
    print(beam_search_transcript)

if __name__ == '__main__':
    PHONE_MAP_PATH = './data/saved_maps.json'
    MODEL_PATH = 'models/phone_model_conv_residual.pth'
    AUDIO_PATH = 'old/old/test_audio/test6.wav'

    meltransform = MelTransform()
    phonetransform = PhoneTransform(PHONE_MAP_PATH)

    # load and set to eval model
    N_CLASSES = config.NUM_OF_PHONE_UNITS + 1
    model = PhoneRecognitionModelResidual(N_CLASSES)
    model = engine.load_model(model, MODEL_PATH)
    model.eval()

    spectrogram = meltransform.build_spectrogram(audio_path=AUDIO_PATH)
    # specs: (ts, n_channels, n_mels) -> (1, 128, ts)
    spectrogram = spectrogram.permute(1, 2, 0)
    spectrogram = (spectrogram - config.MEAN_VALUE) / config.STD_VALUE
    print(spectrogram.size())

    data = spectrogram.unsqueeze(0)

    log_probs = model(data)
    print(log_probs.size())
    # beam_search_ctc_decoder(log_probs)

    _, ids = torch.max(log_probs, dim=-1)
    ids = ids.squeeze(1).numpy().tolist()

    res = phonetransform.decode(ids)
    print(res)