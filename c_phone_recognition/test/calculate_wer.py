from pathlib import Path
import numpy as np


def calculate_wer(reference, hypothesis):
    # Split the reference and hypothesis sentences into words
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    # Initialize asdf matrix with size |ref_words|+1 x |hyp_words|+1
    # The extra row and column are for the case when one of the strings is empty
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    # The number of operations for an empty hypothesis to become the reference
    # is just the number of words in the reference (i.normalize_texts., deleting all words)
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    # The number of operations for an empty reference to become the hypothesis
    # is just the number of words in the hypothesis (i.normalize_texts., inserting all words)
    for j in range(len(hyp_words) + 1):
        d[0, j] = j
    # Iterate over the words in the reference and hypothesis
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            # If the current words are the same, no operation is needed
            # So we just take the previous minimum number of operations
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                # If the words are different, we consider three operations:
                # substitution, insertion, and deletion
                # And we take the minimum of these three possibilities
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)
    # The minimum number of operations to transform the hypothesis into the reference
    # is in the bottom-right cell of the matrix
    # We divide this by the number of words in the reference to get the WER
    wer = d[len(ref_words), len(hyp_words)] / len(ref_words)
    return wer

def open_file(path):
    with open(path, encoding='utf-8') as fp:
        text = fp.read().strip()
        return text

if __name__ == '__main__':
    PREDICTION_DIR = 'C:\\Users\\buryat_saram\\Music\\Buryat Readings\\Testing\\predicted_phones'
    TARGET_DIR = 'C:\\Users\\buryat_saram\\Music\\Buryat Readings\\Testing\\target_phones'

    targets = list(Path(TARGET_DIR).glob('*.txt'))
    predictions = list(Path(PREDICTION_DIR).glob('*.txt'))

    target_seqs = []
    hypothesis_seqs = []

    for target_file, prediction_file in zip(targets, predictions):
        target = open_file(target_file)
        prediction = open_file(prediction_file)

        target_seqs.append(target)
        hypothesis_seqs.append(prediction)

    reference = ' | '.join(target_seqs)
    hypothesis = ' | '.join(hypothesis_seqs)
    print(reference)
    print(hypothesis)
    wer = calculate_wer(reference, hypothesis)

    print('WER:', wer)


