from INPUT import CONTEXT, QUERY
from preprocess import preprocess
import numpy as np

answer = np.fromfile("outputs/output.data", dtype=np.int32)
start = np.asscalar(answer[0])
end = np.asscalar(answer[1])

cw, cc = preprocess(CONTEXT)
print(cw[start:end+1])