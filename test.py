import numpy as np
import operator
a = np.array([[4,2,5],[2,5,7],[6,4,3]])
print(a)
print(np.sort(a,axis=0))
sort = sorted(a,key=operator.itemgetter(0))
print(sort)
print(np.array(sort))