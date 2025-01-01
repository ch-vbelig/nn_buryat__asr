# Params for audio preprocessing
MAX_DURATION = 10  # in seconds
MAX_TARGET_LENGTH = 60  # in seconds
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128

MAX_VALUE = 77.63
MIN_VALUE = -30.66
MEAN_VALUE = -9.52
STD_VALUE = 11.72

# Params for ctc model
HIDDEN_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 1200
LEARNING_RATE = 1e-2
DEVICE = 'cuda'
