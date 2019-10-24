import numpy as np
import pytest

from neural_mnemonic import recommended_search

def test_quick_pi():
    results = recommended_search("314", beam_size=5, vocab_limit=5)
    assert 'meter.' in results
    assert 'mother.' in results

def test_word_penalty():
    long_results = recommended_search("12345", beam_size=3, vocab_limit=5, word_penalty=-4, blank_vocab=1)
    short_results = recommended_search("12345", beam_size=3, vocab_limit=5, word_penalty=4, blank_vocab=1)
    long_result_average_words = np.mean([len(phrase.split()) for phrase in long_results])
    print(f"long result averages {long_result_average_words:.2f}: {long_results}")
    short_result_average_words = np.mean([len(phrase.split()) for phrase in short_results])
    print(f"short result averages {short_result_average_words:.2f}: {short_results}")
    assert long_result_average_words > short_result_average_words * 1.4

def test_char_penalty():
    long_results = recommended_search("89762", beam_size=3, vocab_limit=5, char_penalty=-4, blank_vocab=2)
    short_results = recommended_search("89762", beam_size=3, vocab_limit=5, char_penalty=4, blank_vocab=2)
    long_result_average_words = np.mean([len(phrase) for phrase in long_results])
    print(f"long result averages {long_result_average_words:.2f}: {long_results}")
    short_result_average_words = np.mean([len(phrase) for phrase in short_results])
    print(f"short result averages {short_result_average_words:.2f}: {short_results}")
    assert long_result_average_words > short_result_average_words * 1.5
