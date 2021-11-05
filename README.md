<h1 align="center">Toolkits</h1> <div align="center"> <strong> :hammer_and_wrench: Analysis tools useful for DL experiments :hammer_and_pick: </strong> </div> <br />

Every time I analyze the results of the DL experiment, it is cumbersome to reimplement the analysis functions every time, so I implemented the functions I use frequently. This work is done with [Solang Kim](https://github.com/solangii), and we are practicing a very simple git-flow workflow through this repository.

## Modules

### Visualization

- t-SNE plot

### Clustering quality

- Sum of squared error(SSE) [^1]
- Normalized sum of squared error(nSSE)[^1]
- Intra-class to inter-class variance ratio[^2]

### Pretty print

- Simple print for predictions and true labels

## References

The method presented in the paper below was used.

[^1]: Yoon, Sung Whan, et al. "Xtarnet: Learning to extract task-adaptive representation for incremental few-shot learning." *International Conference on Machine Learning*. PMLR, 2020.
[^2]: Goldblum, Micah, et al. "Unraveling meta-learning: Understanding feature representations for few-shot tasks." *International Conference on Machine Learning*. PMLR, 2020.
[^3]: 파이썬 코딩도장, Unit 45. 모듈과 패키지 만들기. https://dojang.io/mod/page/view.php?id=2448

