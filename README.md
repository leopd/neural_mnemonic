# Neural Number Mnemonic

The [mnemonic major system](https://en.wikipedia.org/wiki/Mnemonic_major_system) is a way to remember long numbers by converting them to phrases.  This code uses a deep-learning language model (GPT2) to find meaningful phrases that correspond to number sequences.

The basics of the code are that each number corresponds to a consonant sound, e.g. "3" corresponds to "M", "1" corresponds to "s" or "t" or "th", and "4" corresponds to "r".  I learned about this system from one of my college math professors, [Dr. Arthur Benjamin](https://math.hmc.edu/benjamin/), who uses it to, among other things, remember intermediate results when performing "mathemagician" tricks like [multiplying 5-digit numbers in his head](https://www.ted.com/talks/arthur_benjamin_does_mathemagic?language=en#t-819057).  To dive into this, I highly recommend his book, [Secrets of Mental Math](https://www.amazon.com/Secrets-Mental-Math-Mathemagicians-Calculation/dp/0307338401).

This code uses a neural-network transformer-based language model (GPT2) from [Huggingface](https://huggingface.co/transformers/) to discriminate between high-quality phrases and low-quality phrases that correspond to any number you provide.  The correspondance isn't perfect, because this code uses letters, while the proper mnemonic is based in pronunciation, and English's spelling/pronunciation is highly irregular.


## Usage

You can run it from command line like this:

```
python neural_mnemonic.py 314159 --beam_size 100
```

This will automatically download the GPT2 models as needed, and after another 30 seconds on a GPU or ~2 minutes on a CPU will find phrases like

```
mother tulip
moderate loop
meteor dolp
```

Larger `beam_size` will find a broader variety of higher quality phrases.


## Requirements

This code requires

* Python 3.6 or higher
* PyTorch
* [Transformers](https://github.com/huggingface/transformers)
* NLTK
* PyTest 

To install you can just

```
pip install -r requirements.txt
```

## Testing

I have included a minimal integration test.  Check out the [source code](test_integration.py) to see how easy it is to write an integration test that gives broad functional coverage.  To test, simply run

```
pytest
```

## Notes on quality, areas to improve

This is far from the best code I've written -- I coded it up in an evening for fun.  I declared success when I saw it recovering the phrase "My turtle poncho" for 31415926.  I've never seen it recover "My turtle poncho, will, my love" perhaps because the phrase itself is somewhat awkward, and thus gets down-voted by GPT2, or perhaps because I haven't included commas in the set of applicable null characters.

It's reasonably fast with a GPU, but result quality would benefit from further speedups.  e.g.  The biggest relatively straightforward speed-up would be achieved by running GPT2 in minibatches to improve GPU efficiency.  Also, I suspect CPU cache hits would go up a lot by scoring the candidates as they're generated (storing the beam in a heeapq) instead of making a giant in-memory list, and then going through it a second time, which kills L2 cache hit rates, but the ~30% de-duplication benefit would be lost. 

Parallelizing the code could improve throughput a ton -- for that I'd probably move the candidate generation inside a PyTorch Dataset, to make use of PyTorch's magic multiprocessing DataLoader, which would also enable minibatches relatively easily too.  But would require redoing a bunch of the beam-search code.

A straightforward improvement would be to turn off the partial-word consideration on the final judgement.  This leads to words like "dolp" showing up in the final results, which are included because it's holding out for the possibility of finding "dolphin".

A complex way to make this work much faster (although I'm not positive you could guarantee quality) would be to use a word-piece approach to the numbers -- building a vocabulary of number-pieces that correspond to valid words in the vocabulary.
