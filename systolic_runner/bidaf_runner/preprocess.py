from INPUT import CONTEXT, QUERY
import os
import numpy as np
import string
import re

def preprocess(text):
   # insert space before and after delimiters, then strip spaces
   delimiters = string.punctuation
   text = re.sub('(['+delimiters+'])', r' \1 ', text).strip()
   # merge consecutive spaces
   text = re.sub('[ ]+', ' ', text)
   # split into lower-case word tokens, in numpy array with shape of (seq, 1)
   words = np.asarray(text.lower().split(' ')).reshape(-1, 1)
   # split words into chars, in numpy array with shape of (seq, 1, 1, 16)
   chars = [[c for c in t][:16] for t in text.split(' ')]
   chars = [cs+['']*(16-len(cs)) for cs in chars]
   chars = np.asarray(chars).reshape(-1, 1, 1, 16)
   return words, chars


if __name__ == "__main__":
    cw, cc = preprocess(CONTEXT)
    qw, qc = preprocess(QUERY)
    # TODO: UNICODE CORRECTNESS. Non-ascii characters crash it for some reason

    cw.tofile(os.path.join('inputs', 'context_word.data'), sep="\n", format="%s")
    cc.tofile(os.path.join('inputs', 'context_char.data'), sep="\n", format="%s")
    qw.tofile(os.path.join('inputs', 'query_word.data'), sep="\n", format="%s")
    qc.tofile(os.path.join('inputs', 'query_char.data'), sep="\n", format="%s")
   
    # C and Q
    np.array([cw.shape[0], qw.shape[0]]).tofile(os.path.join('inputs', 'dims.data'))

    print(cw, cc, qw, qc)

    print(cw.shape)
    print(cc.shape)
    print(qw.shape)
    print(qc.shape)