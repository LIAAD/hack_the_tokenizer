import re
from pathlib import Path

import tqdm

# Import the "SEED" from the "constants" to reproduce results
import sys
sys.path.insert(1, (Path(__file__).parent.parent.resolve() / 'hack_tokenizer/src/utils').as_posix())
from hack_tokenizer.src.utils.constants import SEED, DATA_DIR
from hack_tokenizer.src.benchmark.CalamePT import CalamePT

import numpy as np
np.random.seed(SEED) # Setting numpy seed to reproduce randomness results


def strip_ponctuation_begin_and_end(text):
    # Strip leading non-word characters (e.g., "...Hello" → "Hello")
    text = re.sub(r'^\W+', '', text)
    # Strip trailing non-word characters (e.g., "world!!!" → "world")
    text = re.sub(r'\W+$', '', text)
    return text

def main(num_lines: int=100_000, fertility_output_num_lines: int=4, fertility_boost_num_lines: int=1000):
# -------------------------------------------------------------------------------------------------------
#       Section1: Read data
    # OpenSubtitles dataset obtained from: https://opus.nlpl.eu/results/en&pt/corpus-result-table
    with open(DATA_DIR / 'FULL_opensubtitles_pt-pt.txt', 'r', encoding='utf-8') as f:
        open_subtitles_data = f.readlines()

    # Calamept dataset
    calamept = CalamePT().df[['sentence', 'last_word']].drop_duplicates()
    calamept = (calamept['sentence'] + ' ' + calamept['last_word']).tolist()
# -------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------
#       Metrics Dataset
    selected_lines_metric_eval = np.random.choice(len(open_subtitles_data), size=num_lines, replace=False)
    filename_1 = Path(DATA_DIR) / 'metrics_evaluation_dataset.txt'
    with open(filename_1, 'w', encoding='utf-8') as f:
        f.writelines([strip_ponctuation_begin_and_end(open_subtitles_data[l]) for l in selected_lines_metric_eval if len(open_subtitles_data[l]) > 4])
# -------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------
#       Tokenizer Dataset
    selected_lines_tokenizer = np.random.choice(len(open_subtitles_data), size=num_lines, replace=False)
    filename_2 = Path(DATA_DIR) / 'tokenizer_pt-pt.txt'
    with open(filename_2, 'w', encoding='utf-8') as f:
        f.writelines([strip_ponctuation_begin_and_end(open_subtitles_data[l]) for l in selected_lines_tokenizer if len(open_subtitles_data[l]) > 4])
# -------------------------------------------------------------------------------------------------------
    
# -------------------------------------------------------------------------------------------------------
#       FertilityOutput Dataset (dataset specific for FertilityOutput)
    selected_lines_fertilityOutput = np.random.choice(len(calamept), size=fertility_output_num_lines, replace=False)
    filename_3 = Path(DATA_DIR) / 'fertility_output_evaluation-dataset.txt'
    with open(filename_3, 'w', encoding='utf-8') as f:
        f.writelines([calamept[i].strip() + '\n' for i in selected_lines_fertilityOutput])
# -------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------
#       FertilityBoost Dataset (dataset specific for FertilityOutput)
    selected_lines_fertilityBoost = np.random.choice(len(calamept), size=fertility_boost_num_lines, replace=False)
    filename_3 = Path(DATA_DIR) / 'fertility_boost_evaluation-dataset.txt'
    with open(filename_3, 'w', encoding='utf-8') as f:
        f.writelines([calamept[i].strip() + '\n' for i in selected_lines_fertilityBoost])
# -------------------------------------------------------------------------------------------------------
    return (filename_1, filename_2, filename_3)

if __name__ == '__main__':
    main()