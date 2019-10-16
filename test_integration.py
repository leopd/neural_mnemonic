import pytest

from neural_mnemonic import recommended_search

def test_quick_pi():
    results = recommended_search("314", beam_size=10)
    assert 'meter' in results
    assert 'mature' in results
