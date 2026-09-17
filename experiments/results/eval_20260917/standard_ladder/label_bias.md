# Explicit-label bias

The best constant-label score uses evaluation label frequencies as a diagnostic, not as a fitted predictive model.

| Model | Task | Accuracy | Most predicted label | Share | Best constant label accuracy |
|---|---|---:|---|---:|---:|
| 150m-identcorr | commonsense_qa | 19.49% | A | 98.69% | 20.88% |
| 150m-lite | commonsense_qa | 19.57% | A | 99.43% | 20.88% |
| 150m-postable | commonsense_qa | 19.41% | A | 97.54% | 20.88% |
| 150m-static | commonsense_qa | 19.57% | A | 99.34% | 20.88% |
| 150m-staticcorr | commonsense_qa | 19.49% | A | 98.53% | 20.88% |
| 150m-identcorr | boolq | 58.96% | yes | 91.10% | 62.17% |
| 150m-lite | boolq | 59.30% | yes | 89.05% | 62.17% |
| 150m-postable | boolq | 44.89% | no | 66.70% | 62.17% |
| 150m-static | boolq | 44.53% | no | 71.04% | 62.17% |
| 150m-staticcorr | boolq | 59.79% | yes | 89.66% | 62.17% |
