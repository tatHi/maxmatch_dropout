import random

class MaxMatchTokenizer:
    def __init__(self, vocab=None, midPref='##', headPref=''):
        self.midPref = midPref
        self.headPref = headPref
        self.doNaivePreproc = False
        if vocab:
            self.__build(vocab)

    def __build(self, vocab):
        self.unkToken = '[UNK]'
        self.vocab = set(vocab)
        self.vocab.add(self.unkToken)
        self.vocabSize = len(self.vocab)
        
        self.maxLength = max(len(w) for w in self.vocab)

        self.word2id = {}
        self.id2word = {}
        for w in sorted(self.vocab):
            self.word2id[w] = len(self.word2id)
            self.id2word[self.word2id[w]] = w

    # This function corresponds to Algorithm 1 in the paper.
    def tokenizeWord(self, word, p=0.0):
        subwords = []
        i = 0
        wordLength = len(word)
        while i < wordLength:
            subword = None
            for j in range(1, min(self.maxLength+1, wordLength-i+1)):
                w = word[i:i+j]

                if 0==i: w = self.headPref + w
                if 0<i: w = self.midPref + w

                if w in self.vocab:
                    # random for subword regularization
                    if j==1 or p<random.random():
                        # drop acception with p
                        subword = w
                    
            if subword is None:
                # return unk if including unk
                return [self.unkToken]
            else:
                i += len(subword)-len(self.midPref) if 0<i else len(subword)-len(self.headPref)
                subwords.append(subword)
        return subwords

    def tokenize(self, text, p=0.0):
        if type(text)==list:
            return [self.tokenize(line, p) for line in text]
        if self.doNaivePreproc:
            text = self.naivePreproc(text)
        return [subword for word in text.split() for subword in self.tokenizeWord(word, p)]

    def encode(self, text, p=0.0):
        if type(text)==list:
            return [self.clsTokenId] \
                   + [self.word2id[w] for line in text for w in self.tokenize(line, p)+[self.sepToken]]
        return [self.clsTokenId]+[self.word2id[w] for w in self.tokenize(text, p)]+[self.sepTokenId]

    def loadVocab(self, path):
        words = [line.strip() for line in open(path)]
        self.vocab = set()
        self.word2id = {}
        self.id2word = {}
        for i, w in enumerate(words):
            self.vocab.add(w)
            self.word2id[w] = i
            self.id2word[i] = w
        self.vocabSize = len(self.vocab)
        self.maxLength = max(len(w) for w in self.vocab)
    
        self.unkToken   = '[UNK]'
        self.unkTokenId = self.word2id[self.unkToken]
        self.clsToken   = '[CLS]'
        self.clsTokenId = self.word2id[self.clsToken]
        self.sepToken   = '[SEP]'
        self.sepTokenId = self.word2id[self.sepToken]
 
    def loadBertTokenizer(self, bertTokenizer, doNaivePreproc=False):
        if doNaivePreproc:
            self.doNaivePreproc = doNaivePreproc
            self.bertTokenizer = bertTokenizer

        self.midPref = '##'
        self.vocab = set()
        self.word2id = {}
        self.id2word = {}

        for w, i in bertTokenizer.vocab.items():
            self.vocab.add(w)
            self.word2id[w] = i
            self.id2word[i] = w
        self.vocabSize = len(self.vocab)
        self.maxLength = max(len(w) for w in self.vocab)

        self.unkToken   = bertTokenizer.unk_token
        self.unkTokenId = bertTokenizer.unk_token_id
        self.clsToken   = bertTokenizer.cls_token
        self.clsTokenId = bertTokenizer.cls_token_id
        self.sepToken   = bertTokenizer.sep_token
        self.sepTokenId = bertTokenizer.sep_token_id
        self.bosToken   = bertTokenizer.bos_token
        self.bosTokenId = bertTokenizer.bos_token_id
        self.eosToken   = bertTokenizer.eos_token
        self.eosTokenId = bertTokenizer.eos_token_id

    def naivePreproc(self, text):
        return ' '.join(self.bertTokenizer.tokenize(text)).replace(' '+self.midPref, '')

if __name__=='__main__':
    vocab = '▁a ▁b ▁c abc a b c C S'.split()
    sent = 'aabcb cda'
    print(vocab)
    print(sent)

    mmt = MaxMatchTokenizer(vocab, midPref='', headPref='▁')
    mmt.clsToken = 'C'
    mmt.clsTokenId = mmt.word2id['C']
    mmt.sepToken = 'S'
    mmt.sepTokenId = mmt.word2id['S']

    print(mmt.tokenize(sent))
    print(mmt.encode(sent))

    print(mmt.tokenize([sent, sent]))
    print(mmt.encode([sent, sent]))
