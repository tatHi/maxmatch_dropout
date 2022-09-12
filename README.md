# MaxMatch-Dropout

This is official implementation of MaxMatch-Dropout: Subword Regularization for WordPiece ([arXiv](https://arxiv.org/abs/2209.04126)), which was accepted to appear in COLING2022.
Because the main focus of this paper is to introduce a new subword regularization method named MaxMatch-Dropout, we provide only the tokenizer in this supplementary material.

You can check the tokenization by MaxMatch-Dropout just run the following script with Python3.
The results depend on the random seed because of the probabilistic action in the proposed method.
This example shows a case where we tokenize a word `abcd` with a vocabulary including `{a b c d ab bc cd abc bcd}` (see `runExample.py`).

```
$ python runExample.py
input:
>>> abcd
original tokenization with maximum matching:
>>> ['abc', 'd']
purterbed tokenization with MaxMatch-Dropout:
5 tokenizations with dropout_rate=0.1:
>>> ['abc', 'd']
>>> ['abc', 'd']
>>> ['ab', 'cd']
>>> ['abc', 'd']
>>> ['abc', 'd']
5 tokenizations with dropout_rate=0.2:
>>> ['ab', 'cd']
>>> ['abc', 'd']
>>> ['abc', 'd']
>>> ['ab', 'cd']
>>> ['abc', 'd']
5 tokenizations with dropout_rate=0.3:
>>> ['ab', 'cd']
>>> ['abc', 'd']
>>> ['a', 'bcd']
>>> ['ab', 'cd']
>>> ['abc', 'd']
5 tokenizations with dropout_rate=0.4:
>>> ['abc', 'd']
>>> ['ab', 'cd']
>>> ['abc', 'd']
>>> ['abc', 'd']
>>> ['abc', 'd']
5 tokenizations with dropout_rate=0.5:
>>> ['ab', 'c', 'd']
>>> ['a', 'bc', 'd']
>>> ['ab', 'c', 'd']
>>> ['abc', 'd']
>>> ['ab', 'c', 'd']
5 tokenizations with dropout_rate=0.6:
>>> ['a', 'bcd']
>>> ['ab', 'cd']
>>> ['ab', 'c', 'd']
>>> ['abc', 'd']
>>> ['abc', 'd']
5 tokenizations with dropout_rate=0.7:
>>> ['a', 'b', 'c', 'd']
>>> ['abc', 'd']
>>> ['a', 'b', 'cd']
>>> ['a', 'bc', 'd']
>>> ['abc', 'd']
5 tokenizations with dropout_rate=0.8:
>>> ['a', 'b', 'c', 'd']
>>> ['abc', 'd']
>>> ['ab', 'c', 'd']
>>> ['abc', 'd']
>>> ['a', 'b', 'c', 'd']
5 tokenizations with dropout_rate=0.9:
>>> ['a', 'b', 'c', 'd']
>>> ['a', 'bc', 'd']
>>> ['a', 'b', 'c', 'd']
>>> ['a', 'b', 'c', 'd']
>>> ['a', 'b', 'c', 'd']
5 tokenizations with dropout_rate=1.0:
>>> ['a', 'b', 'c', 'd']
>>> ['a', 'b', 'c', 'd']
>>> ['a', 'b', 'c', 'd']
>>> ['a', 'b', 'c', 'd']
>>> ['a', 'b', 'c', 'd'] 
```

## Usage notes for BertTokenizer(Fast)

You can set the vocabulary of `BertTokenizer` or `BertTokenizerFast` as the following (in the case of using `bert-base-cased`):

```
>>> from transformers import BertTokenizer
>>> import maxMatchTokenizer
>>> tknzr = BertTokenizer.from_pretrained('bert-base-cased')
>>> mmt = maxMatchTokenizer.MaxMatchTokenizer()
>>> mmt.loadBertTokenizer(tknzr)
Using bos_token, but it is not set yet.
Using eos_token, but it is not set yet.
```

This method does NOT load the pre-processing part of `BertTokenizer` or `BertTokenizerFast`.
Thereby, the tokenization of `mmt` results in the different one from the original tokenization of `tknzr`:

```
>>> tknzr.tokenize('Hello, world!')
['Hello', ',', 'world', '!']
>>> mmt.tokenize('Hello, world!')
['Hello', '##,', 'world', '##!']
```

To avoid this problem, **you have to input pre-processed texts to `mmt`**.
For the case of `bert-base-cased`, the following script using `basic_tokenizer` makes `mmt` yield the same tokenization as the original.

```
>>> tmp = ' '.join(tknzr.basic_tokenizer.tokenize('Hello, world!'))
>>> tmp
'Hello , world !'
>>> mmt.tokenize(tmp)
['Hello', ',', 'world', '!']
```

The way of pre-processing varies depending on models, so currently this script does not include these pre-processing.
**When using `BertTokenizer` or `BertTokenizerFast`, please check whether the tokenization of `MaxMatchTokenizer` matches the original tokenization.**

If you do not have time to inspect the detailed pre-processing of the original tokenizer, you can obtain the pre-processed text in a naive but inefficient way: tokenizing and detokenizing the raw text.

```
>>> raw = 'hello, wordpiece!'
>>> tknzr.tokenize(raw)
['hello', ',', 'word', '##piece', '!']
>>> ' '.join(tknzr.tokenize(raw)).replace(' ##', '')
'hello , wordpiece !'
```
