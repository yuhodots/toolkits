## Clustering quality

### sse

> sse(feature, label, opt_print=False, opt_save=False, path=None)

Sum of squared error(SSE)

``` python
from toolkits.utils import load
from toolkits.cluster import sse

feature, classifier, label_f, label_c = load('../dataset/classification300.npz')
result = sse(feature, label_f, opt_print=True, opt_save=True, path='../results/example_sse.csv')
print(result, '\n')

"""
|    `toolkits.cluster.sse`    |
|  Class  |   SSE    |    N    |
--------------------------------
|    0    |  253.52  |   5.0   |
|    1    |  243.27  |   5.0   |
|    2    |  235.84  |   5.0   |
|    3    |  182.7   |   5.0   |
...
|   298   |  236.35  |   5.0   |
|   299   |  235.55  |   5.0   |
--------------------------------
Average SSE: 262.21
262.2054554239909 

"""
```


### batch_sse

> batch_sse(feature, label, opt_print=False, opt_save=False, path=None)

Sum of squared error(SSE) for batch input

``` python
from toolkits.utils import load
from toolkits.cluster import batch_sse

feature, classifier, label_f, label_c = load('../dataset/classification300_batch.npz')
result = batch_sse(feature, label_f, opt_print=True, opt_save=True, path='../results/example_batch_sse.csv')
print(result, '\n')

"""
| `toolkits.cluster.batch_sse` |
|  Class  |   SSE    |    N    |
--------------------------------
|    0    |  253.52  |   5.0   |
|    1    |  243.27  |   5.0   |
|    2    |  235.84  |   5.0   |
...
|   298   |  236.35  |   5.0   |
|   299   |  235.55  |   5.0   |
--------------------------------
Average SSE: 262.21
262.2054554239909 

"""
```


### nsse

> nsse(feature, label, opt_print=False, opt_save=False, path=None)

SSE normalized by the squared distance to the nearest interfering centroid(nSSE)

``` python
from toolkits.utils import load
from toolkits.cluster import nsse

feature, classifier, label_f, label_c = load('../dataset/classification300.npz')
result = nsse(feature, label_f, opt_print=True, opt_save=True, path='../results/example_nsse.csv')
print(result, '\n')

"""
|   `toolkits.cluster.nsse`    |
|  Class  |   nSSE   |    N    |
--------------------------------
|    0    |   6.75   |   5.0   |
|    1    |   9.51   |   5.0   |
|    2    |   9.27   |   5.0   |
...
|   298   |   8.68   |   5.0   |
|   299   |  10.89   |   5.0   |
--------------------------------
Average nSSE: 8.77
8.767743170300136 

"""
```


### batch_nsse

> batch_nsse(feature, label, opt_print=False, opt_save=False, path=None)

SSE normalized by the squared distance to the nearest interfering centroid(nSSE) for batch input

``` python
from toolkits.utils import load
from toolkits.cluster import batch_nsse

feature, classifier, label_f, label_c = load('../dataset/classification300_batch.npz')
result = batch_nsse(feature, label_f, opt_print=True, opt_save=True, path='../results/example_batch_nsse.csv')
print(result, '\n')

"""
|`toolkits.cluster.batch_nsse` |
|  Class  |   nSSE   |    N    |
--------------------------------
|    0    |   6.75   |   5.0   |
|    1    |   9.51   |   5.0   |
|    2    |   9.27   |   5.0   |
...
|   298   |   8.68   |   5.0   |
|   299   |  10.89   |   5.0   |
--------------------------------
Average nSSE: 8.77
8.767743170300136 

"""
```

### nearc

> nearc(feature, label, N=5, opt_print=False)

Top N nearest interfering centroid

``` python
from toolkits.utils import load
from toolkits.cluster import nearc

feature, classifier, label_f, label_c = load('../dataset/classification300.npz')
nearc(feature, label_f, 5, opt_print=True)

"""
  Class  | Near1 | Near2 | Near3 | Near4 | Near5 
--------------------------------------------------
    0    |  179     124      41     279     190   
    1    |  253     268     263      2      254   
    2    |  131      37     263      1       9  
    3    |  135     103     155     298     282   ...
"""
```

### rfc

> rfc(feature, label)

Feature space clustering quality(R_fc)

``` python
from toolkits.utils import load
from toolkits.cluster import rfc

feature, classifier, label_f, label_c = load('../dataset/classification300.npz')
print(rfc(feature, label_f))    # = 361.05813174016123
```
