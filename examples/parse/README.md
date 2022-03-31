## Log parser

### between_lines

> between_lines(input_str, beg, end, single=False)

Extract the log between the two input sentences

```python
from toolkits.parse import between_lines

input_str = "Hello\nWorld\n!!\nHello\nWorld\n!!"
list_data = between_lines(input_str, beg="Hello", end="!!")

"""
list_data = [['Hello', 'World'], ['Hello', 'World']]
"""
```

### between_lines_on_file

> between_lines_on_file(file_path, beg, end, single=False)

Extract the log between the two input sentences on the target file

```python
from toolkits.parse import between_lines_on_file

list_data = between_lines_on_file('../dataset/document1.txt', beg="Organizational", end="Finance", single=True)
	for item in list_data:
		print("\n".join(item))
      
"""
Organizational structure
GitHub, Inc. was originally a flat organization with no middle managers; in other words, "everyone is a manager" (self-management).[16] Employees could choose to work on projects that interested them (open allocation), but salaries were set by the chief executive.[17][needs update]
In 2014, GitHub, ...
"""
```

### between_lines_on_dir

> between_lines_on_dir(dir_path, beg, end, single=False, path_filter=None, make_file=False, result_file='result.txt')

Extract the log between the two input sentences on the target directory

```python
from toolkits.parse import between_lines_on_dir

dict_data = between_lines_on_dir('../dataset', beg="Organizational", end="Finance", single=True, path_filter="document")

"""
dict_data = {
	"../dataset/document3.txt": [],
	"../dataset/document2.txt": [],
	"../dataset/document1.txt": [['Organizational structure', 'GitHub, Inc. was originally a flat organization with no middle managers; in other words, "everyone is a manager" (self-management).[16] Employees could choose to work on projects that interested them (open allocation), but salaries were set by the chief executive.[17][needs update]', 'In 2014, GitHub, Inc. introduced a layer of middle management amid harassment claims made against senior management. Tom Preston-Werner resigned as CEO amid the scandal.[18]', '']],
}
"""
```

```python
from toolkits.parse import between_lines_on_dir

dict_data = between_lines_on_dir('../dataset', beg="Organizational", end="Finance", single=True, path_filter="document", make_file=True, result_file="../results/result.txt")

for k, v in dict_data.items():
	print(k)
  	for item in v:
    	print("\n".join(item))

"""
../dataset/../dataset/document3.txt
../dataset/../dataset/document2.txt
../dataset/../dataset/document1.txt
Organizational structure
GitHub, Inc. was originally a flat organization with no middle managers; in other words, "everyone is a manager" (self-management).[16] Employees could choose to work on projects that interested them (open allocation), but salaries were set by the chief executive.[17][needs update]
In 2014, GitHub, ...
"""
```

