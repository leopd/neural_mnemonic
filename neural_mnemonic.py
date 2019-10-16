"""Find phrases corresponding to numbers using the Major mnemonic system.
Uses GPT2 neural transformer language model to evaluate the quality of phrases.
Phrases are generated using the most popular words for each pronunciation,
as translated by the CMU US English Dictionary.
"""

import argparse
import heapq
import json
import functools
import random
import torch
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Tuple


def numberstr(numbers:str) -> str:
    """Validator/constructor for argparse that confirms a string only has numbers in it.
    """
    for digit in numbers:
        if digit not in "0123456789":
            raise argparse.ArgumentTypeError("Please only include digits 0-9 without punctuation")
    return numbers


class NextBeamDataset(torch.utils.data.IterableDataset):
    """An iterable dataset meant to be used only once, which generates all the
    candidates for the next beam.

    Pattern of use should be: iterate over it with a DataLoader, score the results,
    and then call register_beam_winners.
    """

    def __init__(self, numbers:str, mnd:dict, blank_vocab:int=0):
        self.beam = [('', numbers)]
        self.next_beam = []
        self.mnd = mnd
        self.longest_num = max(set([len(nums) for nums in self.mnd.keys()]))
        self.blank_vocab = blank_vocab

    def longest_left(self) -> int:
        numbers_length = set()
        for words_so_far, digits_to_go in self.beam:
            numbers_length.add(len(digits_to_go))
        return max(numbers_length)

    def __iter__(self):
        """Goes through the current beam, and yields a new set of candidates.
        Ones which are ahead, we store for the next epoch.
        Yields dicts with 2 keys: 'phrase', 'numtogo'
        """
        # First, count how far they are.  Only candidates the furthest behind get to advance.
        target_length = self.longest_left()
        if target_length == 0:
            print("Nothing left to do!")
            return
        print(f"This time, we're targeting {target_length} length sequences")

        # Now start to generate the next set of candidates
        next_beam = set()
        self.carry_forward = []
        for words_so_far, digits_to_go in self.beam:
            if not len(digits_to_go) == target_length:
                self.carry_forward.append((words_so_far, digits_to_go))
            else:
                if digits_to_go:
                    # This one needs to be expanded.
                    yield from self._expand_candidate(words_so_far, digits_to_go, self.blank_vocab)
        print(f"Carried forward {len(self.carry_forward)} shorter entries")

    def _expand_candidate(self, words_so_far:str, digits_to_go:str, blank_vocab:int):
        """Yields a set of beam-entries to evaluate.
        """
        if blank_vocab:
            # Recurse to add all the possible blank words
            for blank_word in self.mnd[''][:blank_vocab]:
                new_phrase = words_so_far + " " + blank_word
                yield from self._expand_candidate(new_phrase, digits_to_go, 0)
            # pass through to allow non-blanks as well

        # Now consume as many digits_to_go as possible, down to 1
        max_digits = min(self.longest_num, len(digits_to_go))
        assert max_digits > 0  # something's wrong
        for dig_length in range(max_digits,0,-1):
            numbers_to_convert = digits_to_go[:dig_length]
            new_digits_to_go = digits_to_go[dig_length:]
            if not numbers_to_convert in self.mnd:
                # This set of numbers doesn't have any valid words in the vocab.
                continue
            if new_digits_to_go:
                suffix = ''
            else:
                # This one is at the end.  Treat it like a complete sentence.
                suffix = '.'
            for word in self.mnd[numbers_to_convert]:
                yield {
                    'phrase': words_so_far + " " + word + suffix,
                    'numtogo': new_digits_to_go,
                }

    def register_beam_winners(self, winners:list):
        """Records the winning candidates from the last epoch, to make
        the foundation for the next.  This also carries forward everything
        from the last epoch which wasn't explicitly evaluated
        """
        self.beam = self.carry_forward
        for phrase, numbers in winners:
            assert isinstance(phrase,str)
            assert isinstance(numberstr(numbers),str)
            self.beam.append((phrase, numbers))


class SentenceScorer():
    """Computes a language model score for a sentence.
    """

    def __init__(self, model_name:str='distilgpt2'):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        if torch.cuda.is_available():
            self._cuda = True
            self.model.cuda()
        else:
            self._cuda = False
        self.model.eval()

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.eos = self.tokenizer.encode(self.tokenizer.special_tokens_map['eos_token'])


    @functools.lru_cache(maxsize=10000000)
    def perplexity(self, text:str) -> float:
        """Returns perplexity of a sentence.  Lower is better.
        """
        # Based on https://github.com/huggingface/transformers/issues/1009
        tokens = self.tokenizer.tokenize(text)
        tensor_input = torch.tensor([self.eos + self.tokenizer.encode(tokens)])
        with torch.no_grad():
            if self._cuda:
                tensor_input = tensor_input.to('cuda')
            outputs = self.model(tensor_input, labels=tensor_input)
            loss, logits = outputs[:2]
        return float(loss)*len(tokens)  # total perplexity
        #return float(loss)  # per-token perplexity, favors unusual words with lots of tokens. 
        # e.g. when using per-token perplexity, every 3 wants to be "Yamaha".

    def __call__(self, text:str) -> float:
        return self.perplexity(text)


class BestHeap():
    """Remembers the best (lowest) N scores seen so far.
    Uses a max-heap so the worst score (largest number) can always be removed quickly.
    (Since python implements min-heap by default, we have to invert the score internally.)

    Restores numbers to their original value when you iterate over them.
    """

    def __init__(self, size:int):
        self.h = []
        self.size = size

    def add(self, entry, score):
        # add some float64 noise to avoid sorting collisions, since the "entry" is typically a dict, which
        # python can't compare.
        fuzzed_score = float(score) * (1 + 1e-8 * random.random())  
        heapitem = (-fuzzed_score, entry)
        if len(self.h) < self.size:
            heapq.heappush(self.h, heapitem)
        else:
            heapq.heappushpop(self.h, heapitem)

    def flush(self) -> List[Tuple[str, float]]:
        """Empties the heap, and returns the contents in order.
        """
        out = []
        while self.h:
            negscore, entry = heapq.heappop(self.h)
            out.insert(0, (entry, -negscore))
        return out



class BeamSearcher():

    def __init__(self, numbers:str, mnd:dict, beam_size:int, scorer:SentenceScorer, word_penalty:float, 
                 char_penalty:float, blank_vocab:int=0):
        self.numbers = numbers
        self.total_len = len(numbers)
        assert self.total_len > 0
        self.scorer = scorer
        self.data = NextBeamDataset(numbers, mnd, blank_vocab)
        self.beam_size = beam_size
        self.winners = None
        self.word_penalty = word_penalty
        self.char_penalty = char_penalty

    def do_beam(self):
        best = BestHeap(self.beam_size)
        print("Generating candidates")
        candidates = list(self.data)
        for candidate in tqdm(candidates, "scoring"):
            score = self.calc_score(candidate)
            best.add(candidate, score)
        self.winners = best.flush()
        winners_without_scores = [(rec['phrase'], rec['numtogo']) for rec, score in self.winners]
        self.data.register_beam_winners(winners_without_scores)

    def calc_score(self, candidate:dict) -> float:
        raw_score = self.scorer(candidate['phrase'])
        numbers_used = self.total_len - len(candidate['numtogo'])
        raw_score /= numbers_used  # perplexity goes up by length, so normalize to the fraction of the way through the problem
        word_cnt = len(candidate['phrase'].split())
        score = raw_score * (word_cnt ** self.word_penalty)
        char_cnt = len(candidate['phrase'])
        score *= (char_cnt ** self.char_penalty)
        return score

    def run(self, max_out:int=100):
        while self.data.longest_left() > 0:
            self.do_beam()
            self.summarize_results()
        out = self.final_scores(max_out)
        return out

    def final_scores(self, max_out:int):
        phrases = set()
        for rec, score in self.winners:
            phrases.add(rec['phrase'])
        for phrase, numbers in self.data.beam:
            phrases.add(phrase)
        best = BestHeap(self.beam_size)
        for phrase in phrases:
            phrase = phrase.strip()
            score = self.scorer(phrase) 
            best.add(phrase, score)
        print("\nFinal results...")
        final_results = list(best.flush())
        out = []
        for phrase, score in final_results[:max_out]:
            print(f"{score: 8.2f} {phrase}")
            out.append(phrase)
        return out

    def summarize_results(self):
        print("This round:")
        for rec, score in self.winners[:10]:
            print(f"{score: 8.2f} {rec['phrase']}")

def shrink_dict(mnemonic_dict:dict, vocab_limit:int) -> dict:
    """Takes a dict which maps numberstr -> List[word-strings], and
    limits each value so that it has at most vocab_limit entries.  If the dict
    is sorted (like the one on disk is), this keeps the most common words.
    """
    out = {}
    for numbers, wordlist in mnemonic_dict.items():
        if wordlist:
            out[numbers] = wordlist[:vocab_limit]
    return out
        

def recommended_search(numbers:str, max_out:int=100, gpt:str='distilgpt2', beam_size:int=50, 
                       vocab_limit:int=20, blank_vocab:int=0, word_penalty:float=0.3, 
                       char_penalty:float=0):
    scorer = SentenceScorer(gpt)
    mnemonic_dict = json.load(open("mnemonic-dict.json","rt"))
    reduced_dict = shrink_dict(mnemonic_dict, vocab_limit)
    bs = BeamSearcher(numbers, reduced_dict,  beam_size, scorer, word_penalty, char_penalty, blank_vocab)
    out = bs.run(max_out=max_out)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('numbers', 
        type=numberstr, 
        help='The sequence of numbers to find a phrase for')
    parser.add_argument('-b', '--beam_size',
        type=int,
        default=20,
        help='Number of phrases to keep in the beam search. Larger is slower, better.')
    parser.add_argument('-wp', '--word_penalty',
        type=float,
        default=0.5,
        help='How much to discourage phrases with lots of words. Try -1ish to 1ish. Larger means shorter phrases.')
    parser.add_argument('-cp', '--char_penalty',
        type=float,
        default=0.1,
        help='How much to discourage phrases with lots of characters. Try -1ish to 1ish. Larger means shorter phrases.')
    parser.add_argument('-v', '--vocab_limit',
        type=int,
        default=25,
        help='Max number of words to consider for each number combination. Larger is slower, better.')
    parser.add_argument('-bv', '--blank_vocab',
        type=int,
        default=1,
        help='How many "blank" words (like "a" which have no number) to consider between each numeric word. Can be 0, default 1, larger is significantly slower.')
    parser.add_argument('-mo', '--max_out',
        type=int,
        default=100,
        help='Maximum number of phrases to display at end.')
    parser.add_argument('--gpt',
        default='distilgpt2',
        choices=['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large', 'check-if-huggingface-added-more'],
        help='Which pre-trained GPT2 model to use to judge phrase quality')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    recommended_search(**dict(args.__dict__))
