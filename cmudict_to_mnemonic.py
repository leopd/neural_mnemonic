"""Converts the CMU US English Dictionary to the Mnemonic major system.

Ref https://github.com/cmusphinx/cmudict
https://en.wikipedia.org/wiki/Mnemonic_major_system

Before starting, first download the dict with

wget https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict

(Note that this utility outputs in a random order.  The vocab_limit option
works much better if the words are sorted by unigram frequency, which 
this code doesn't capture.
"""

from collections import defaultdict
import json


def pronunciation_to_numbers(pronunciation:str) -> str:
    sounds = pronunciation.split()

    REJECT = set(['NG'])  # too ambiguous to include.
    for not_allowed in REJECT:
        if not_allowed in sounds:
            return '!'

    MNEMONIC_CODE = {
        'S': 0,
        'Z': 0,
        'D': 1,
        'T': 1,
        'TH': 1,
        'DH': 1,
        'N': 2,
        'M': 3,
        'R': 4,
        'ER0': 4,
        'ER1': 4,
        'ER2': 4,
        'L': 5,
        'SH': 6,
        'JH': 6,
        'CH': 6,
        'ZH': 6,
        'G': 7,
        'K': 7,
        'F': 8,
        'V': 8,
        'B': 9,
        'P': 9,
    }
    outstr = ''
    for sound in sounds:
        if sound in MNEMONIC_CODE:
            outstr += str(MNEMONIC_CODE[sound])
    return outstr


def load_cmudict_as_dict(filename:str="cmudict.dict") -> dict:
    """Returns a python dict with key as the word, and value as a list of pronunciations.
    """
    raw = open(filename, "rt").readlines()
    cmu_dict = {}
    for line in raw:
        word, pronunciation = line.rstrip().split(maxsplit=1)
        if "(" in word:
            base_word = word[:word.index('(')]
            cmu_dict[base_word].append(pronunciation)
        else:
            cmu_dict[word] = [pronunciation]
    return cmu_dict


def invert_dict(cmu_dict:dict) -> dict:
    """Converts to a dict with keys as number-strings, and values as lists of words.
    """
    number_word_map = defaultdict(list)
    saved_words = 0
    for word, pronunciations in cmu_dict.items():
        possible_numbers = set()
        for pronunciation in pronunciations:
            numbers = pronunciation_to_numbers(pronunciation)
            possible_numbers.add(numbers)
        if len(possible_numbers) > 1:
            print(f"Word {word} is ambiguous. Could be {possible_numbers}")
        else:
            only_number = list(possible_numbers)[0]
            if only_number == '!':
                # Ambiguous sound, don't include.
                continue
            number_word_map[only_number].append(word)
            saved_words += 1
    print(f"Of the {len(cmu_dict)} words, {saved_words} are unambiguous in their mnemonic")
    return dict(number_word_map)


def main(outfile:str="mnemonic-dict.json"):
    cmu_dict = load_cmudict_as_dict()
    number_map = invert_dict(cmu_dict)
    with open(outfile, "wt") as f:
        f.write(json.dumps(number_map, indent=2))
    print(f"Saved {len(number_map)} entries to {outfile}")

if __name__ == "__main__":
    main()

