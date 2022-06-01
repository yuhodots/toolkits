## Linear Algebra

### get_singular_values

> get_singular_values(feature, label)

 Get singular values per class

``` python
from toolkits.utils import load
from toolkits.linalg import get_singular_values

feature, classifier, label_f, label_c = load('../dataset/classification300.npz')
print(get_singular_values(feature, label_f))

"""
{
    '0': [10.557849, 7.639909, 7.019653, 5.865659, 2.8754462e-06],
    '1': [9.532024, 8.322581, 6.824369, 6.0476246, 2.1209382e-06],
    ...
    '299': [9.742858, 7.4412527, 6.8818126, 6.1556606, 1.3703436e-06],
}
"""
```

### get_sum_of_singular_values

> get_sum_of_singular_values(feature, label)

Get sum of singular values per class

``` python
from toolkits.utils import load
from toolkits.linalg import get_sum_of_singular_values

feature, classifier, label_f, label_c = load('../dataset/classification300.npz')
print(get_sum_of_singular_values(feature, label_f))

"""
{
    '0': 31.083072676776737, '1': 30.726601337399416, '2': 30.28996430842676, ...
}
"""
```

### get_average_sum_of_singular_values

> get_average_sum_of_singular_values(feature, label)

Get average of sum of singular values per class

``` python
from toolkits.utils import load
from toolkits.linalg import get_average_sum_of_singular_values

feature, classifier, label_f, label_c = load('../dataset/classification300.npz')
print(get_average_sum_of_singular_values(feature, label_f))     # 31.572425177141117
```
