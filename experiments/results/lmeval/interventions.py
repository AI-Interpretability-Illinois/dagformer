"""Evaluation-only routing substitutions using separately calibrated tables."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from interp_common import flatten_alpha, unflatten_alpha
from interp_editing import layer_chunks


MODES = ("none", "pred_position", "pred_global", "corr_zero", "corr_position",
         "corr_global", "pred_position_corr_position", "context_pred", "context_corr", "context_both")


def intervention_label(mode, gamma=1.25):
    return mode + (f"_gamma{gamma:g}" if mode.startswith("context_") else "")


def install_context_edit(model, mode, circuit_path, gamma):
    from eval_context_fidelity import apply_edges, read_circuit
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    edges = read_circuit(circuit_path, 10)
    if any(layer >= L or head >= H for layer, stream, head, source in edges):
        raise ValueError("The fixed context-fidelity circuit does not fit this architecture")
    channel = mode.removeprefix("context_")
    chunks = layer_chunks(L, H)
    handles = []
    if channel in ("pred", "both"):
        def pred_hook(module, inputs, output):
            flat = flatten_alpha(output)
            edited = [apply_edges(flat[..., lo:hi], i + 1, H, edges, gamma)
                      for i, (lo, hi) in enumerate(chunks)]
            return unflatten_alpha(torch.cat(edited, -1), L, H)
        handles.append(model.fourway_predictor.register_forward_hook(pred_hook))
    if channel in ("corr", "both"):
        for index, mlp in enumerate(model.fourway_model.correction_mlps):
            def corr_hook(module, inputs, output, layer=index + 1):
                return apply_edges(output, layer, H, edges, gamma)
            handles.append(mlp.register_forward_hook(corr_hook))
    model._routing_eval_handles = handles
    return {"mode": mode, "gamma": gamma, "circuit_source": circuit_path,
            "edges": edges, "scope": "fixed head-mean-deviation edit on predictor/corrections"}


def install_intervention(model, mode, table_path, *, circuit_path=None, gamma=1.25):
    if mode == "none":
        return None
    if not hasattr(model, "fourway_model"):
        raise ValueError("Routing interventions require a FourWay checkpoint")
    if mode.startswith("context_"):
        return install_context_edit(model, mode, circuit_path, gamma)
    pred_mode = "position" if mode.startswith("pred_position") else (
        "global" if mode == "pred_global" else None)
    corr_mode = "position" if mode.endswith("corr_position") else (
        mode.removeprefix("corr_") if mode.startswith("corr_") else None)
    cfg = model.config
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    payload, tables = {}, {}
    if mode != "corr_zero":
        if not table_path:
            raise ValueError("This routing intervention requires --routing-table")
        payload = torch.load(table_path, map_location="cpu", weights_only=False)
        if (payload["L"], payload["H"]) != (L, H):
            raise ValueError("Routing-table dimensions disagree with the checkpoint")
        tables = {k: v.to(model.device) for k, v in payload["tables"].items()}

    def expand(table, batch, length):
        if table.shape[0] != 1 and length > table.shape[0]:
            raise ValueError("The calibrated routing table is shorter than this context")
        return table[:length].unsqueeze(0).expand(batch, length, -1)

    handles = []
    if pred_mode:
        def pred_hook(module, inputs, output):
            batch, length = inputs[0].shape
            flat = expand(tables["pred_" + pred_mode], batch, length)
            return unflatten_alpha(flat, L, H)
        handles.append(model.fourway_predictor.register_forward_hook(pred_hook))
    if corr_mode:
        chunks = layer_chunks(L, H)
        for index, mlp in enumerate(model.fourway_model.correction_mlps):
            def corr_hook(module, inputs, output, index=index):
                if corr_mode == "zero":
                    return torch.zeros_like(output)
                start, end = chunks[index]
                flat = expand(tables["corr_" + corr_mode], *output.shape[:2])
                return flat[..., start:end].to(output.dtype)
            handles.append(mlp.register_forward_hook(corr_hook))
    # Keep handles attached for the lifetime of this evaluation instance.
    model._routing_eval_handles = handles
    return {"mode": mode, "table": str(table_path) if table_path else None,
            "calibration": {k: v for k, v in payload.items() if k != "tables"},
            "scope": "post-training inference substitution; predictor still executes before hook",
            "parameter_count": "original trained checkpoint, including the substituted predictor"}
