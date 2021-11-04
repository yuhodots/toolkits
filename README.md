# Toolkits
A repository that implements the analysis tools required for DL ​​experiments

Every time I analyze the results of the DL experiment, it is cumbersome to reimplement the analysis functions every time, so I implemented the functions I use frequently. The list of functions is as follows.

## Visualization

- t-SNE plot

## Clustering quality analysis

- Sum of squared error(SSE) [^1]
- Normalized sum of squared error(nSSE)[^1]
- Intra-class to inter-class variance ratio[^2]

## Pretty printer

- Simple print for predictions and true labels

## References

The method presented in the paper below was used.

[^1]: Yoon, Sung Whan, et al. "Xtarnet: Learning to extract task-adaptive representation for incremental few-shot learning." *International Conference on Machine Learning*. PMLR, 2020.
[^2]: Goldblum, Micah, et al. "Unraveling meta-learning: Understanding feature representations for few-shot tasks." *International Conference on Machine Learning*. PMLR, 2020.
