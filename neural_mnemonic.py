"""Find phrases corresponding to numbers using the Major mnemonic system.
Uses GPT2 neural transformer language model to evaluate the quality of phrases,
along with NLTK corpus to validate word correctness.

Considerably faster with a CUDA GPU.  Increasing beam_size will increase 
quality of results, but also increase time.  
Setting non_numeric_count>1 increases search time dramatically.
"""

import argparse
import functools
import torch
from nltk.corpus import words
import numpy as np
from tqdm.auto import tqdm
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer


CODE = {
    0: ['s', 'z', 'ss'],
    1: ['t', 'tt', 'd', 'th'],
    2: ['n'],
    3: ['m'],
    4: ['r'],
    5: ['l', 'll'],
    6: ['sh', 'ch', 'j', 'dg'],
    7: ['c', 'k', 'ck'],
    8: ['f', 'v', 'ph'],
    9: ['b', 'p'],
    None: ['a','e','i','o','u','w','y',' ', ''],
}
#NOTE: this is intrinsically approximate/flawed because of English's irregular spelling.

class SentenceScorer():
    """Computes a language model score for a sentence.
    """

    def __init__(self, model_name:str='gpt2', expand:int=2):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        if torch.cuda.is_available():
            self._cuda = True
            self.model.cuda()
        else:
            self._cuda = False
        self.model.eval()

    @functools.lru_cache(maxsize=10000000)
    def perplexity(self, text:str) -> float:
        """Returns perplexity of a sentence.  Lower is better.
        """
        # Based on https://github.com/huggingface/transformers/issues/1009
        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
        tokenize_input = self.tokenizer.tokenize(text)
        #50256 is the token_id for <|endoftext|>
        tensor_input = torch.tensor([ [50256]  +  self.tokenizer.convert_tokens_to_ids(tokenize_input)])
        with torch.no_grad():
            if self._cuda:
                tensor_input = tensor_input.to('cuda')
            outputs = self.model(tensor_input, labels=tensor_input)
            loss, logits = outputs[:2]
        return float(loss)*len(tokenize_input)

    def __call__(self, text:str) -> float:
        #return self.perplexity(text) / len(text.split())
        return self.perplexity(text)

class NltkWordScorer():
    """Just says if each word is a word or not, using nltk corpus.
    Sentence score is the fraction of words that are in the corpus.
    For the final word in the sentence, it allows partial words -- that
    is word prefixes that could be completed into full words like "transpor"
    """

    def __init__(self):
        self._vocab = set(words.words())
        self._prefixes = set()
        for word in self._vocab:
            for i in range(len(word)):
                sub = word[:i]
                self._prefixes.add(sub)

    @functools.lru_cache(maxsize=10000000)
    def __call__(self, text:str) -> float:
        words = text.split()
        bad = 0
        refset = self._prefixes  
        for word in words[::-1]:  # go backwards
            if not word in refset:
                bad += 1
            refset = self._vocab  # last word is checked against prefixes, rest against vocab
        return bad / len(words)


class CompositeScorer():

    def __init__(self, word_scorer, backup_scorer, factor:float=100):
        self._word = word_scorer
        self._backup = backup_scorer
        self.factor = factor

    @functools.lru_cache(maxsize=10000000)
    def __call__(self, text:str) -> float:
        base = self._word(text)
        if base == 0:
            full_score = self._backup(text)
            # Since last word could be incomplete, see what the big scorer says without it.
            all_but_last_word = ' '.join(text.split()[:-1]) + '.'
            all_but_score = self._backup(all_but_last_word) 
            return (full_score + all_but_score) / self.factor
        else:
            return base * self.factor


class BeamSearcher():

    def __init__(self, scorer:SentenceScorer, beam_size:int, expand_cnt:int=2):
        self.scorer = scorer
        self.beam = beam_size
        self._candidates = []
        self.EXPANDED_CODE = {}
        for d in range(10):
            self.EXPANDED_CODE[d] = self._expand_code_nones(d, cnt_nones=expand_cnt)


    def _expand_code1(self, digit:int, base:[str]=['']) -> [str]:
        out = []
        for c in CODE[digit]:
            for b in base:
                out.append(b+c)
        return out

    def _expand_code_nones(self, digit:int, base:[str]=[''], cnt_nones:int=2) -> [str]:
        out = base
        for _ in range(cnt_nones):
            out = self._expand_code1(None, out)
        out = self._expand_code1(digit, out)
        for _ in range(cnt_nones):
            out = self._expand_code1(None, out)
        return list(set(out))

    def convert(self, digits:str, max_out:int=100) -> [str]:
        """Converts a string of digits to the most likely sentence.
        """
        self._candidates = ['']
        for i, digit in enumerate(digits):
            sub = digits[:(i+1)]
            print(f"Adding digit {digit} up to {sub}")
            d = int(digit)
            self._next_digit(d)
            self._beam_down()
            print(f"Best so far for {sub}... {self._candidates[:10]}")
            print(f"Worst is number {len(self._candidates)}: '{self._candidates[-1]}' with score {self.scorer(self._candidates[-1]):.4f}")
        return self._candidates[:max_out]

    def _next_digit(self, digit:int):
        next_set = []
        for base in tqdm(self._candidates, "generating candidates"):
            next_set += self._expand_possibilities(base, digit)
        self._candidates = self._condense_list(next_set)

    def _condense_list(self, candidates:[str]) -> [str]:
        print(f"Deduplicating set of {len(candidates)}")
        deduped = set(candidates)
        round2 = set()
        for s in deduped:
            s = s.strip()
            s = s.replace("  "," ")
            round2.add(s)
        print(f"Deduplicated down to {len(round2)}")
        return list(round2)

    def _beam_down(self):
        scores = {}
        for candidate in tqdm(self._candidates, "scoring"):
            scores[candidate] = self.scorer(candidate)
        print("Sorting...")
        ordered = sorted(scores.items(), key=lambda kv: kv[1])
        self._candidates = [kv[0] for kv in ordered[:self.beam]]

    def _expand_possibilities(self, base:str, digit:int) -> [str]:
        out = [base + ec for ec in self.EXPANDED_CODE[digit]]
        return out


def recommended_search(numbers:str, max_out:int=100, beam_size:int=500, non_numeric_count:int=1, gpt:str='distilgpt2'):
    fancy = SentenceScorer(gpt)
    cheap = NltkWordScorer()
    both = CompositeScorer(cheap, fancy)
    bs = BeamSearcher(both, beam_size=beam_size, expand_cnt=non_numeric_count)
    out = bs.convert(numbers, max_out=max_out)
    return out


def check_number_string(numbers:str) -> str:
    for digit in numbers:
        if digit not in "0123456789":
            raise argparse.ArgumentTypeError("Please only include digits 0-9 without punctuation")
    return numbers

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('numbers', 
        type=check_number_string, 
        help='The sequence of numbers to find a phrase for')
    parser.add_argument('-b', '--beam_size',
        type=int,
        default=500,
        help='Number of phrases to keep in the beam search')
    parser.add_argument('-nn', '--non_numeric_count',
        type=int,
        default=1,
        help='Number of non-mnemonic characters to try before and after each number')
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
    out = recommended_search(args.numbers, args.max_out, args.beam_size, args.non_numeric_count, args.gpt)
    print("Final results:\n")
    for result in out:
        print(result)
    
