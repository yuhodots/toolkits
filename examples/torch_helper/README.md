## PyTorch helper function

### freeze_selected_param

> freeze_selected_param(model, target)

Freeze the weights with the selected name

``` python
freeze_list = ['bn1', 'bn2', 'fc']
freeze_selected_param(model, target=freeze_list)
```

### get_important_param_idx

> get_important_param_idx(model, ratio, inverse=False)

Get important parameters indices

``` python
idx = self.get_session_trainable_param_idx(model, 0.1)
```
