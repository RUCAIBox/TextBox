# Configuration of Search Space of Hyper-parameters Tuning

Create a text file and define parameters to be tuned line by line in format below:

```text
<label> <algorithm> <space>
```

- `<label>`: Parameters to be optimized
- `<algorithm>`: Algorithm to define search space, like `choice` and `loguniform`.
- `<space>`: Search space. Any Python object within one line is supported.

Search space should be defined accordingly to algorithm:

- `choice`: `<space>` receives **an iterable of choices**.
- `loguniform`: `<space>` receives **an iterable of positional arguments (low, high)**, which returns a value drawn according to $exp(uniform(low, high))$.
- more algorithm visit [parameter expressions](http://hyperopt.github.io/hyperopt/getting-started/search_spaces/) for a full support list.

### Example

The example used in tutorial: [`hyperopt_example.test`](textbox/properties/hyperopt_example.test)

```text
learning_rate loguniform (-8, 0)
embedding_size choice [64, 96 , 128]
train_batch_size choice [512, 1024, 2048]
```
