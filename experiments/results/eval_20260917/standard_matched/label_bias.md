# Explicit-label bias

The best constant-label score uses evaluation label frequencies as a diagnostic, not as a fitted predictive model.

| Model | Task | Accuracy | Most predicted label | Share | Best constant label accuracy |
|---|---|---:|---|---:|---:|
| 150m-baseline | commonsense_qa | 19.74% | A | 99.02% | 20.88% |
| 150m-dagformer | commonsense_qa | 19.66% | A | 99.43% | 20.88% |
| 300m-baseline | commonsense_qa | 19.74% | A | 98.94% | 20.88% |
| 300m-dagformer | commonsense_qa | 19.57% | A | 99.59% | 20.88% |
| 600m-baseline | commonsense_qa | 19.74% | A | 99.43% | 20.88% |
| 75m-baseline | commonsense_qa | 19.57% | A | 100.00% | 20.88% |
| 75m-dagformer | commonsense_qa | 19.57% | A | 99.92% | 20.88% |
| 150m-baseline | boolq | 40.92% | no | 87.80% | 62.17% |
| 150m-dagformer | boolq | 50.67% | yes | 52.35% | 62.17% |
| 300m-baseline | boolq | 56.27% | yes | 77.89% | 62.17% |
| 300m-dagformer | boolq | 60.80% | yes | 94.22% | 62.17% |
| 600m-baseline | boolq | 59.36% | yes | 91.50% | 62.17% |
| 75m-baseline | boolq | 37.95% | no | 98.84% | 62.17% |
| 75m-dagformer | boolq | 42.48% | no | 72.60% | 62.17% |
