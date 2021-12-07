<h1 align="center">DL Toolkits</h1> <div align="center"> <strong> :man_technologist:&nbsp; Analytical tools useful for deep learning experiments &nbsp;:woman_technologist: </strong></div><br/>
<div align="center"><img src="https://img.shields.io/badge/version-1.0-green.svg">&nbsp;<img src="https://img.shields.io/badge/LICENSE-TBD-orange.svg"></div><br/>

Whenever I analyzed the results of the DL experiment, I had to re-implement the analysis function every time. So, I implement frequently used functions in this repository. New features continue to be implemented, and simple examples of function usages can be found in the `examples` directory.



## Installation

You can install the package with `pip` command. Python>=3 are supported.

```
pip install dl-toolkits
```

You can check the version of the package using the following commands.

```python
import toolkits
print(toolkits.__version__)
```

## Modules

### Visualization

- [`viz.tsne`](https://github.com/yuhodots/toolkits/blob/main/toolkits/viz/tsne.py#L20): t-SNE plot

### Clustering quality

- [`cluster.sse`](https://github.com/yuhodots/toolkits/blob/main/toolkits/cluster/sse.py#L36): Sum of squared error(SSE) [^1]
- [`cluster.nsse`](https://github.com/yuhodots/toolkits/blob/main/toolkits/cluster/nsse.py#L48): SSE normalized by the squared distance to the nearest interfering centroid(nSSE) [^1]

### Pretty print

- [`pprint.pred`](https://github.com/yuhodots/toolkits/blob/main/toolkits/pprint/pred.py#L44): Simple print for predictions and true labels

### PyTorch helper function

- [`torch_helper.freeze_selected_param`](): Freeze the weight with the selected name
- [`torch_helper.get_important_param_idx`](): Get the important parameters indices (large absolute value)[^2]

## References

[^1]: Yoon, Sung Whan, et al. "Xtarnet: Learning to extract task-adaptive representation for incremental few-shot learning." *International Conference on Machine Learning*. PMLR, 2020.
[^2]: Mazumder, Pratik, Pravendra Singh, and Piyush Rai. "Few-Shot Lifelong Learning." *Proceedings of the AAAI Conference on Artificial Intelligence*. Vol. 35. No. 3. 2021.
