MAX_LENGTH = 10
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128

HIDDEN_SIZE = 128
# BATCH_SIZE = 32
BATCH_SIZE = 8
EPOCHS = 200
LEARNING_RATE = 10
DEVICE = 'cuda'


ECODER_SAVE_PATH = './model/encoder.pth'
DECODER_SAVE_PATH = './model/decoder.pth'
ATTN_DECODER_SAVE_PATH = './model/attn_decoder.pth'