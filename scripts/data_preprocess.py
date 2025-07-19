import re
from pathlib import Path

import tqdm

# Import the "SEED" from the "constants" to reproduce results
import sys
sys.path.insert(1, (Path(__file__).parent.parent.resolve() / 'hack_tokenizer/src/utils').as_posix())
from constants import SEED, DATA_DIR

import numpy as np
np.random.seed(SEED) # Setting numpy seed to reproduce randomness results


def strip_ponctuation_begin_and_end(text):
    return re.sub(r'(?<!\w)[^\w]+|\B[^\w]+(?!\w)', '', text)

def get_random_lines(file_path, N, encoding='utf-8'):
    # Step 1: Get total lines and byte offsets of each line
    offsets = []
    with open(file_path, 'rb') as f:
        offsets.append(0)  # First line starts at byte 0
        pbar = tqdm.tqdm(desc='Scanning file')
        while f.readline():
            offsets.append(f.tell())  # Record byte offsets of each line
            pbar.update(1)
        total_lines = len(offsets) - 1  # Adjust for 0-based indexing
        pbar.close()

    # Step 2: Randomly select N unique line numbers
    selected_indices = np.random.choice(total_lines, size=min(N, total_lines), replace=False)
    selected_indices.sort()  # Optimize sequential reading

    # Step 3: Fetch selected lines using seek()
    selected_lines = []
    with open(file_path, 'r', encoding=encoding) as f:
        for idx in selected_indices:
            f.seek(offsets[idx])
            selected_lines.append(f.readline().strip())

    return selected_lines

def main(num_lines: int=100_000):
    # OpenSubtitles dataset obtained from: https://opus.nlpl.eu/results/en&pt/corpus-result-table
    dataset = Path(DATA_DIR) / 'FULL_opensubtitles_pt-pt.txt'

    # Pick 10_000 random lines from the file
    # lines = get_random_lines(dataset, 10_000)
    file = open(dataset, 'r')
    lines = file.readlines()
    file.close()
    selected_lines_metric_eval = np.random.choice(len(lines), size=num_lines, replace=False)

    # Write lines to a new file
    filename = Path(DATA_DIR) / 'metrics_evaluation_dataset.txt'
    with open(filename, 'w') as f:
        f.writelines([strip_ponctuation_begin_and_end(lines[l]) for l in selected_lines_metric_eval if len(lines[l]) > 4])

    selected_lines_tokenizer = np.random.choice(len(lines), size=num_lines, replace=False)
    # Write lines to a new file
    filename = Path(DATA_DIR) / 'tokenizer_pt-pt.txt'
    with open(filename, 'w') as f:
        f.writelines([strip_ponctuation_begin_and_end(lines[l]) for l in selected_lines_tokenizer if len(lines[l]) > 4])

    return filename

if __name__ == '__main__':
    main()