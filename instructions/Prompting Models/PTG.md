## PTG

First, you may copy the[ `prompt_source.pth` ](https://github.com/RUCAIBox/Transfer-Prompts-for-Text-Generation/blob/main/prompt_source.pth)into the TextBox folder (*i.e.*, <your clone dir>/TextBox).

you can conduct the cross-dataset experiments on cnndm dataset using this command:

```bash
python run_textbox.py --model=PTG --dataset=cnndm --model_path=facebook/bart-large
```

In this default case, the source tasks (datasets) is msn, mn, and nr.

You can use `--dataset=xxx` to specify the dataset name.

In addition, you can also specify the source tasks using `--source_task=list_of_task`. The default setting is equivalent to `--source_task=\[\'msn\',\'mn\',\'nr\'\]`.

There are also several cases used in the paper:

- `--source_task=cross-dataset2`: tc, da, mw.
- `--source_task=cross-task1`: squad, wiki, quora, wp, cnndm.
- `--source_task=cross-task2`: squad, wiki, quora, wp, pc.