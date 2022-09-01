import maxMatchTokenizer

inputWord = 'abcd'

# Vocab
vocab = 'a b c d ab bc cd abc bcd'.split()
tknzr = maxMatchTokenizer.MaxMatchTokenizer(vocab, midPref='', headPref='')

# Deterministic Tokenization
print('input:')
print('>>>', inputWord)
print('original tokenization with maximum matching:')
print('>>>', tknzr.tokenize(inputWord))

# Probabilistic Tokenization with MaxMatch-Dropout
print('purterbed tokenization with MaxMatch-Dropout:')

for dropout_rate in range(1, 11):
    dropout_rate *= 0.1
    print('5 tokenizations with dropout_rate=%.1f:'%dropout_rate)
    for i in range(5):
        print('>>>', tknzr.tokenize(inputWord, dropout_rate))
