import numpy as np
import glob
import os
import pickle
import lmdb
import pyarrow
import fasttext
from loguru import logger
from scipy import linalg


class Vocab:
    PAD_token = 0
    SOS_token = 1
    EOS_token = 2
    UNK_token = 3

    def __init__(self, name, insert_default_tokens=True):
        self.name = name
        self.trimmed = False
        self.word_embedding_weights = None
        self.reset_dictionary(insert_default_tokens)

    def reset_dictionary(self, insert_default_tokens=True):
        self.word2index = {}
        self.word2count = {}
        if insert_default_tokens:
            self.index2word = {self.PAD_token: "<PAD>", self.SOS_token: "<SOS>",
                               self.EOS_token: "<EOS>", self.UNK_token: "<UNK>"}
        else:
            self.index2word = {self.UNK_token: "<UNK>"}
        self.n_words = len(self.index2word)  # count default tokens

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def add_vocab(self, other_vocab):
        for word, _ in other_vocab.word2count.items():
            self.index_word(word)

    # remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('    word trimming, kept %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # reinitialize dictionary
        self.reset_dictionary()
        for word in keep_words:
            self.index_word(word)

    def get_word_index(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.UNK_token

    def load_word_vectors(self, pretrained_path, embedding_dim=300):
        print("  loading word vectors from '{}'...".format(pretrained_path))

        # initialize embeddings to random values for special words
        init_sd = 1 / np.sqrt(embedding_dim)
        weights = np.random.normal(0, scale=init_sd, size=[self.n_words, embedding_dim])
        weights = weights.astype(np.float32)

        # read word vectors
        word_model = fasttext.load_model(pretrained_path)
        for word, id in self.word2index.items():
            vec = word_model.get_word_vector(word)
            weights[id] = vec
        self.word_embedding_weights = weights

def build_vocab(name, data_path, cache_path, word_vec_path=None, feat_dim=None):
    print('  building a language model...')
    lang_model = Vocab(name)
    print('    indexing words from {}'.format(data_path))
    index_words_from_textgrid(lang_model, data_path)

    if word_vec_path is not None:
        lang_model.load_word_vectors(word_vec_path, feat_dim)
    else:
        print('    loaded from {}'.format(cache_path))
        with open(cache_path, 'rb') as f:
            lang_model = pickle.load(f)
        if word_vec_path is None:
            lang_model.word_embedding_weights = None
        elif lang_model.word_embedding_weights.shape[0] != lang_model.n_words:
            logging.warning('    failed to load word embedding weights. check this')
            assert False

    with open(cache_path, 'wb') as f:
        pickle.dump(lang_model, f)

    return lang_model

def index_words(lang_model, data_path):
    #index words form text
    with open(data_path, "r") as f:
        for line in f.readlines():
            line = line.replace(",", " ")
            line = line.replace(".", " ")
            line = line.replace("?", " ")
            line = line.replace("!", " ")
            for word in line.split(): 
                lang_model.index_word(word)
    print('    indexed %d words' % lang_model.n_words)

def index_words_from_textgrid(lang_model, data_path):
    import textgrid as tg 
    trainvaltest=os.listdir(data_path)
    for loadtype in trainvaltest:
        if "." in loadtype: continue #ignore .ipynb_checkpoints
        texts = os.listdir(data_path+loadtype+"/text/")
        for textfile in texts:
            tgrid = tg.TextGrid.fromFile(data_path+loadtype+"/text/"+textfile)
            for word in tgrid[0]:
                word_n, word_s, word_e = word.mark, word.minTime, word.maxTime
                word_n = word_n.replace(",", " ")
                word_n = word_n.replace(".", " ")
                word_n = word_n.replace("?", " ")
                word_n = word_n.replace("!", " ")
                #print(word_n)
                lang_model.index_word(word_n)
    print('    indexed %d words' % lang_model.n_words)    
    
if __name__ == "__main__":
    #11195 for all, 5793 for 4 speakers
    build_vocab("beat_english_15_141", "/home/ma-user/work/datasets/beat_cache/beat_english_15_141/", "/home/ma-user/work/datasets/beat_cache/beat_english_15_141/vocab.pkl", "/home/ma-user/work/datasets/cc.en.300.bin", 300)
    