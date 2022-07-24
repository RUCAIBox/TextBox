**This is a temporary file that needs to be merged.** 

### W&B tutorial

#### Login

You only need to register and log in to W&B by pasting the API key retrieved from [here](https://wandb.ai/authorize) for the first run.

You can also log in manually beforehand from command line: `wandb login`

If you are running your code in jupyter environments (like colab), you may want to login by simply setting a environment variable:

```python
%env WANDB_API_KEY=<your-key>
```

#### Disable

If you are debugging your model, you may want to disable W&B with `wandb disabled` in the command line and **none of the metrics** will be recorded.
To re-enable it, use `wandb enabled`.

You can also disable **sync only** with `wandb offline` and enable it again with `wandb online`. 
The local files can be uploaded by executing `wandb sync`.

For more information, use `wandb --help` or visit [Weights&Biases](https://docs.wandb.ai/).
