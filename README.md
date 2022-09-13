# MaxMatch-Dropout

This is official implementation of MaxMatch-Dropout: Subword Regularization for WordPiece ([arXiv](https://arxiv.org/abs/2209.04126)), which was accepted to appear in COLING2022.
Because the main focus of this paper is to introduce a new subword regularization method named MaxMatch-Dropout, we provide only the tokenizer in this repository.

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

## Use for BertTokenizer

You can set the vocabulary of `BertTokenizer`, `BertTokenizerFast`, or `BertJapaneseTokenizer` as the following (in the case of using `bert-base-cased`):

```
>>> from transformers import BertTokenizer
>>> import maxMatchTokenizer
>>> tknzr = BertTokenizer.from_pretrained('bert-base-cased')
>>> mmt = maxMatchTokenizer.MaxMatchTokenizer()
>>> mmt.loadBertTokenizer(tknzr, doNaivePreproc=True)
Using bos_token, but it is not set yet.
Using eos_token, but it is not set yet.
```

And you can tokenize and encode texts similar to the original `BertTokenizer`.

```
>>> text = 'hello, wordpiece!'
>>> tknzr.tokenize(text)
['hello', ',', 'word', '##piece', '!']
>>> mmt.tokenize(text)
['hello', ',', 'word', '##piece', '!']
>>> tknzr.encode(text)
[101, 19082, 117, 1937, 9641, 106, 102]
>>> mmt.encode(text)
[101, 19082, 117, 1937, 9641, 106, 102]
```

Both `tokenize()` and `encode` supports MaxMatch-Dropout, which can be tuned with a dropout rate `p`.


```
>>> mmt.tokenize(text, p=0.5)
['h', '##ello', ',', 'word', '##pie', '##ce', '!']
>>> mmt.tokenize(text, p=0.5)
['hello', ',', 'w', '##or', '##d', '##pie', '##c', '##e', '!']
>>> mmt.tokenize(text, p=0.5)
['hello', ',', 'w', '##ord', '##piece', '!']
>>> mmt.encode(text, p=0.5)
[101, 19082, 117, 192, 6944, 9641, 106, 102]
>>> mmt.encode(text, p=0.5)
[101, 19082, 117, 1937, 8508, 10294, 1162, 106, 102]
>>> mmt.encode(text, p=0.5)
[101, 19082, 117, 1937, 9641, 106, 102]
```

With `doNaivePreproc=True`, `MaxMatchTokenizer` first calls the original `BertTokenizer` to tokenize and detokenize texts.
This is because we want to use the same pre-processing as the original tokenizer but the pre-processing varies depending on each model and we cannot support all of them.
This naive pre-processing does not guarantee that MaxMatchTokenizer can use the completely the same pre-processing as the original one.
But this works well in many cases.

This naive processing is, of course, inefficient in terms of processing speed because this calls the tokenization process twice (BertTokenizer's tokenization and MaxMatchTokenizer's tokenization).
Please see the following section to know the situation when loading `BertTokenizer` without the `doNaivePreproc` option.

## Usage notes for BertTokenizer

When calling `loadBertTokenizer()`, `doNaivePreproc` is set as `False` in default of specification.

```
>>> from transformers import BertTokenizer
>>> import maxMatchTokenizer
>>> tknzr = BertTokenizer.from_pretrained('bert-base-cased')
>>> mmt = maxMatchTokenizer.MaxMatchTokenizer()
>>> mmt.loadBertTokenizer(tknzr)          // doNaivePreproc=False
Using bos_token, but it is not set yet.
Using eos_token, but it is not set yet.
```

In this case, `mmt` does not handle any pre-processing before tokenization.
Thereby, the tokenization of `mmt` results in the different one from the original tokenization of `tknzr`:

```
>>> tknzr.tokenize('Hello, world!')
['Hello', ',', 'world', '!']
>>> mmt.tokenize('Hello, world!')
['Hello', '##,', 'world', '##!']  // different from the original tokenization
```

**When using `BertTokenizer` or `BertTokenizerFast`, please check whether the tokenization of `MaxMatchTokenizer` matches the original tokenization.**

To avoid this problem of different tokenization (without using the `doNaivePreproc` option), **you have to input pre-processed texts to `mmt`**.
For the case of `bert-base-cased`, the following script using `basic_tokenizer` makes `mmt` yield the same tokenization as the original.

```
>>> tmp = ' '.join(tknzr.basic_tokenizer.tokenize('Hello, world!'))
>>> tmp
'Hello , world !'
>>> mmt.tokenize(tmp)
['Hello', ',', 'world', '!']
```


If you do not have time to inspect the detailed pre-processing of the original tokenizer, you can obtain the pre-processed text in a naive but inefficient way: tokenizing and detokenizing the raw text.

```
>>> raw = 'hello, wordpiece!'
>>> tknzr.tokenize(raw)
['hello', ',', 'word', '##piece', '!']
>>> ' '.join(tknzr.tokenize(raw)).replace(' ##', '')
'hello , wordpiece !'
```

And this process is also implemented in `mmt.naivePreproc()`, which is called with `doNaivePreproc=True`.

```
>>> from transformers import BertTokenizer
>>> import maxMatchTokenizer
>>> tknzr = BertTokenizer.from_pretrained('bert-base-cased')
>>> mmt = maxMatchTokenizer.MaxMatchTokenizer()
>>> mmt.loadBertTokenizer(tknzr, doNaivePreproc=True)
Using bos_token, but it is not set yet.
Using eos_token, b
>>> mmt.naivePreproc('hello, wordpiece!')
'hello , wordpiece !'
```
