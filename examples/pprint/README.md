## Pretty print

### pred_summary

> pred_summary(feature, classifier, label_f, label_c, similarity='euclidean', opt_save=False, path=None)

``` python
from toolkits.utils import load
from toolkits.pprint import pred_summary

feature, classifier, label_f, label_c = load('../dataset/classification300.npz')
pred_summary(feature, classifier, label_f, label_c, opt_save=True, path='example_pred.csv')

"""
  Class  |    N    | Correct | Predict 
---------------------------------------
    0    |    5    |    3    |  41      0      0     172     0   
    1    |    5    |    3    |  285    72      1      1      1   
    2    |    5    |    2    |   9     131     2     90      2   
    3    |    5    |    5    |   3      3      3      3      3   
    4    |    5    |    4    |   4     152     4      4      4   ...
"""
```
