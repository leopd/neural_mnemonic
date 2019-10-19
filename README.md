# Neural Number Mnemonic

The [mnemonic major system](https://en.wikipedia.org/wiki/Mnemonic_major_system) is a way to remember long numbers by converting them to phrases.  This code uses a deep-learning language model (GPT2) to find meaningful phrases that correspond to number sequences.

The basics of the code are that each number corresponds to a consonant sound, e.g. "3" corresponds to "M", "1" corresponds to "S" or "T" or "TH", and "4" corresponds to "R".  I learned about this system from one of my college math professors, [Dr. Arthur Benjamin](https://math.hmc.edu/benjamin/), who uses it to, among other things, remember intermediate results when performing "mathemagician" tricks like [multiplying 5-digit numbers in his head](https://www.ted.com/talks/arthur_benjamin_does_mathemagic?language=en#t-819057).  To dive into this, I highly recommend his book, [Secrets of Mental Math](https://www.amazon.com/Secrets-Mental-Math-Mathemagicians-Calculation/dp/0307338401).

This code uses a neural-network transformer-based language model (GPT2) from [Huggingface](https://huggingface.co/transformers/) to discriminate between high-quality phrases and low-quality phrases that correspond to any number you provide.  The correspondance isn't perfect, because this code uses letters, while the proper mnemonic is based in pronunciation, and English's spelling/pronunciation is highly irregular.


## Usage

You can run it from command line like this:

```
python neural_mnemonic.py 31415
```

This will automatically download the GPT2 models as needed, and after another 30 seconds on a GPU or ~1.5 minutes on a CPU will find phrases corresponding to 31415 like

```
moderately
meter tall
mother doll
```

There are a number of tuning parameters you can adjust to trade-off time vs quality in the results.  Run with `--help` for a description.

If you leave it running overnight, it can discover great sentences for long sequences.  For example, the standard major mneumonic for the first 24 digits of pi is **"my turtle poncho will my love pickup my new mover ginger"** which is a fine sentence.  To find viable alternatives, try running it like:

```
# This will take several hours
python neural_mnemonic.py 314159265358979323846264 --beam_size 100 --word_penalty -0.4 --blank_vocab 10 --vocab_limit 50
```

* **a moderately high pendulum will have a big boom in my average injury.**
* **a mother who would help you in a huge way while you may leave a big poem in my fridge in a jar.**


## Requirements

This code requires

* Python 3.6 or higher
* PyTorch 1.3  (required for IterableDataset)
* [Transformers](https://github.com/huggingface/transformers)
* PyTest 

To install you can just

```
pip install -r requirements.txt
```

## Testing

I have included a few integration tests.  Check out the [source code](test_integration.py) to see how easy it is to write robust functional tests that give broad functional coverage.  To test, simply run

```
pytest
```

## Notes on code, vocab, areas to improve

This is a nearly complete rewrite from the first version which added characters together hoping to find words.  The insolvable problem with that approach was the English spelling/pronunciation is irregular.  Now we have a full english vocabulary aligned to number sequences, derived from the [Carnegie Mellon University US English Dictionary](https://github.com/cmusphinx/cmudict) which includes pronunciations, and an English unigram frequency table from Google for sorting the words so that when you apply the `vocab_limit` it picks the most popular words for each number combo.  I've removed any words which would be ambiguous in their pronunciation such that they might imply different number sequences -- e.g. "twenty" could correspond to "12" or "121" depending on whether you pronounce the second "t" or not, so it's not an option for either number combo.  I've also removed any words with the "ing" sound because it's not clear how they fit into the mnemonic code.  This process was mostly automatic, in `cmudict_to_mnemonic.py` but I manually edited the "blank" words (at the top of [`mnemonic-dict.json`](mnemonic-dict.json) because these have a huge impact on the algorithm.

The beam search is a little non-standard in that the candidates can be of different lengths, consuming different numbers of digits from the input.  I thus normalize the phrase's NLP perplexity by the number of digits consumed in order to provide a fair comparison.  Also this means the different parts of the beam don't move ahead together.  I opt to expand the beam as needed.  This seems to work fairly well.  How the perplexity gets normalized has a really big effect on the style of the output, thus leading to a number of control hyper-parameters.  By default GPT2 normalizes by token, which if left alone heavily favors proper-noun words which the word-piece model divides into lots of tokens.  e.g. in this mode, the number "3" tends to always end up as "Yamaha" which I'm guessing is 3 tokens in GPT2, but only a single number.  GPT2 naturally has no problem with things that look like long proper names, so the results end up pretty ridiculous.  So I start with the total perplexity of the whole phrase, normalize by the number of digits used, and then add penalties for the number of words or characters, as configured in the hyper-parameters.  These length penalties interact meaningfully with the number of "blank" words (words without any numeric representation such as "a" and "you") that can be inserted into the phrase to make the sentences more natural, as exposed by the `blank_vocab` control, which can really slow the calculation down if set too high (I wouldn't go over 10).

It's reasonably fast with a GPU, but result quality would benefit from further speedups since you could search more.  The biggest relatively straightforward speed-up would be achieved by running GPT2 in minibatches to improve GPU utilization, as the GPU seems to be the bottleneck, surprise surprise.  This should be fairly easy now that we're using a torch datasets to generate the candidates, I just haven't done it yet, because of sequence padding hassle.

The LRU cache on GPT2 evaluations provided by the `@functools.lru_cache` annotation can chew up gigabytes of memory (I'm guessing not a problem for most machines these days).  It probably doesn't help at all for single (command-line) runs, but can dramatically speed up repeated tinkering in a notebook -- very similar runs can be nearly instantaneous.  I should include an example of doing this.

I'd like to add some diversity into the beam search candidates, perhaps following a technique like [Diverse Beam Search, Vijayakumar, et al 2016](https://arxiv.org/pdf/1610.02424.pdf), and expose controls to force it to dig into a particular beam candidate.  I'd also like to let it figure out where to end sentences, and explore multi-sentence phrases.
