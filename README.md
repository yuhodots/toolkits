<h1 align="center">Toolkits</h1> <div align="center"> <strong> :hammer_and_wrench: Analysis tools useful for DL experiments :hammer_and_pick: </strong> <br/> <img src="https://img.shields.io/badge/version-1.0.0-green.svg"><img src="https://img.shields.io/badge/LICENSE-TBD-orange.svg"></div>  <br/>

Every time I analyze the results of the DL experiment, it is cumbersome to reimplement the analysis functions every time, so I implemented the functions I use frequently. This work is done with [Solang Kim](https://github.com/solangii) :raised_hands:

## Installation

You can install the package with the `pip` command. Python>=3 are supported.

```
pip install dl-toolkits
```

You can check the version of the package.

```python
import toolkits
print(toolkits.__version__)
```

## Modules

### Visualization

- t-SNE plot

### Clustering quality

- Sum of squared error(SSE) [^1]
- SSE normalized by the squared distance to the nearest interfering centroid(nSSE) [^1]
- Measurements of feature clustering(R_fc)[^2]
- Measurements of hyperplane variation(R_hv)[^2]

### Pretty print

- Simple print for predictions and true labels

## References

The method presented in the paper below was used.

[^1]: Yoon, Sung Whan, et al. "Xtarnet: Learning to extract task-adaptive representation for incremental few-shot learning." *International Conference on Machine Learning*. PMLR, 2020.
[^2]: Goldblum, Micah, et al. "Unraveling meta-learning: Understanding feature representations for few-shot tasks." *International Conference on Machine Learning*. PMLR, 2020.
