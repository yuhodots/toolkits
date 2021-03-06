<h1 align="center">DL Toolkits</h1> <div align="center"> <strong> :man_technologist:&nbsp; Analytical tools for deep learning experiments &nbsp;:woman_technologist: </strong></div><br/>
<div align="center"><img src="https://img.shields.io/badge/version-1.1-green.svg">&nbsp;<img src="https://img.shields.io/badge/LICENSE-MIT License-orange.svg"></div><br/>

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

- [`viz.tsne`](https://github.com/yuhodots/toolkits/tree/main/examples/viz#tsne): t-SNE plot

### Clustering quality

- [`cluster.sse`](https://github.com/yuhodots/toolkits/tree/main/examples/cluster#sse): Sum of squared error(SSE) [^1]
- [`cluster.batch_sse`](https://github.com/yuhodots/toolkits/tree/main/examples/cluster#batch_sse): Sum of squared error(SSE) for batch input
- [`cluster.nsse`](https://github.com/yuhodots/toolkits/tree/main/examples/cluster#nsse): SSE normalized by the squared distance to the nearest interfering centroid(nSSE) [^1]
- [`cluster.batch_nsse`](https://github.com/yuhodots/toolkits/tree/main/examples/cluster#batch_nsse): SSE normalized by the squared distance to the nearest interfering centroid(nSSE) for batch input
- [`cluster.nearc`](https://github.com/yuhodots/toolkits/tree/main/examples/cluster#nearc): Top N nearest interfering centroid
- [`cluster.rfc`](https://github.com/yuhodots/toolkits/tree/main/examples/cluster#rfc): Feature space clustering quality(R_fc) [^2]

### Linear algebra

- [`linalg.get_singular_values`](https://github.com/yuhodots/toolkits/tree/main/examples/linalg#get_singular_values): Get singular values per class
- [`linalg.get_sum_of_singular_values`](https://github.com/yuhodots/toolkits/tree/main/examples/linalg#get_sum_of_singular_values): Get sum of singular values per class
- [`linalg.get_average_sum_of_singular_values`](https://github.com/yuhodots/toolkits/tree/main/examples/linalg#get_average_sum_of_singular_values): Get average of sum of singular values per class [^3]

### Pretty print

- [`pprint.pred_summary`](https://github.com/yuhodots/toolkits/tree/main/examples/pprint#pred_summary): Simple print for predictions and true labels

### Log parser

- [`parse.between_lines`](https://github.com/yuhodots/toolkits/tree/main/examples/parse#between_lines): Extract the log between the two input sentences
- [`parse.between_lines_on_file`](https://github.com/yuhodots/toolkits/tree/main/examples/parse#between_lines_on_file): Extract the log between the two input sentences on the target file
- [`parse.between_lines_on_dir`](https://github.com/yuhodots/toolkits/tree/main/examples/parse#between_lines_on_dir): Extract the log between the two input sentences on the target directory

### PyTorch helper function

- [`torch_helper.freeze_selected_param`](https://github.com/yuhodots/toolkits/tree/main/examples/torch_helper#freeze_selected_param): Freeze the weights with the selected name
- [`torch_helper.get_important_param_idx`](https://github.com/yuhodots/toolkits/tree/main/examples/torch_helper#get_important_param_idx): Get important parameters indices[^4]

## References

[^1]: Yoon, Sung Whan, et al. "Xtarnet: Learning to extract task-adaptive representation for incremental few-shot learning." *International Conference on Machine Learning*. PMLR, 2020.
[^2]: Goldblum, Micah, et al. "Unraveling meta-learning: Understanding feature representations for few-shot tasks." *International Conference on Machine Learning*. PMLR, 2020.
[^3]: Verma, Vikas, et al. "Manifold mixup: Better representations by interpolating hidden states." *International Conference on Machine Learning*. PMLR, 2019.
[^4]: Mazumder, Pratik, Pravendra Singh, and Piyush Rai. "Few-Shot Lifelong Learning." *Proceedings of the AAAI Conference on Artificial Intelligence*. Vol. 35. No. 3. 2021.

Pedregosa, Fabian, et al. "Scikit-learn: Machine learning in Python." *the Journal of machine Learning research* 12 (2011): 2825-2830.
