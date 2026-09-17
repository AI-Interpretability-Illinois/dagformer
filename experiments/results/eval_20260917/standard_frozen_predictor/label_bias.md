# Explicit-label bias

The best constant-label score uses evaluation label frequencies as a diagnostic, not as a fitted predictive model.

| Model | Task | Accuracy | Most predicted label | Share | Best constant label accuracy |
|---|---|---:|---|---:|---:|
| 150m-dagformer__pred_position | commonsense_qa | 19.66% | A | 99.43% | 20.88% |
| 300m-dagformer__pred_position | commonsense_qa | 19.57% | A | 99.59% | 20.88% |
| 75m-dagformer__pred_position | commonsense_qa | 19.57% | A | 99.92% | 20.88% |
| 150m-dagformer__pred_position | boolq | 50.76% | yes | 52.69% | 62.17% |
| 300m-dagformer__pred_position | boolq | 60.86% | yes | 94.34% | 62.17% |
| 75m-dagformer__pred_position | boolq | 42.26% | no | 72.57% | 62.17% |
