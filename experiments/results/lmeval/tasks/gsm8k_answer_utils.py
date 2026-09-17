"""Score only gold solution bytes, conditioned on the GSM8K question."""


def answer_target(doc):
    # Match the suffix of gsm8k_bpb's full Question/Answer text exactly.
    return " " + doc["answer"]


def process_results(doc, results):
    loglikelihood, _ = results[0]
    return {"bits_per_byte": (loglikelihood, len(answer_target(doc).encode("utf-8")))}
