## Visualization

### tsne

> tsne(feature, classifier, label_f, label_c, save_path, perplexity=30, seed=42)

``` python
from examples.dataset.pseudo_cluster import ClusterData
from toolkits.viz import tsne

data = ClusterData()
feature, classifier, label_f, label_c = data()
tsne(feature, classifier, label_f, label_c, 'tsne.png')
```