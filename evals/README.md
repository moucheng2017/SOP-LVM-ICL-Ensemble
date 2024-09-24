# Evaluations

This directory contains scripts to run auto-evaluations of model generated standard operating procedures (SOPs).

1. Set up your openai api key first in the terminal:
```
export OPENAI_API_KEY="your-key"
```

2. Change the paths in the file 'eval.py' from line 229 to line 231. Then run the eval.py in the folder evals.

<!-- - [`eval.py`](./eval.py) - The main script used to run auto-evaluations of the model's generated SOPs against the "gold standard" SOPs. Change from line 230 in eval.py for your own use. -->

```
python eval.py
```
