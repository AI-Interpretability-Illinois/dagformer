# KV-cache generation check

The same 300M baseline checkpoint is evaluated with and without KV caching.
The cached full-test run is restricted to the uncached run's document IDs.
Document and prompt hashes, checkpoint, and generation settings must match.

| Endpoint | Cached | Uncached | Difference (percentage points) | Paired 95% interval |
|---|---:|---:|---:|---|
| strict-match | 0.00% | 0.00% | +0.00 | [+0.00, +0.00] |
| flexible-extract | 2.50% | 2.50% | +0.00 | [+0.00, +0.00] |

Full response text changes on 24/200 documents; the extracted answer changes on 11.
Mean repeated whitespace-token 4-gram fraction is 0.861 with caching and 0.857 without.

Intervals describe this document subset, without training-seed uncertainty.
