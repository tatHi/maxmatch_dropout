# MaxMatch-Dropout

This is official implementation of `MaxMatch-Dropout: Subword Regularization for WordPiece`, which was accepted to appear in COLING2022.
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
