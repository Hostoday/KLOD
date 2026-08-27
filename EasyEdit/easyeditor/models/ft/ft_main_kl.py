from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook

from .ft_hparams import FTHyperParams


FT_LOSS_OBJECTIVE_ALIASES = {
    "standard": "standard_ft",
    "standard_ft": "standard_ft",
    "ce": "standard_ft",
    "ft": "standard_ft",
    "kmtd": "kmtd_prefix_kl",
    "kmtd_prefix_kl": "kmtd_prefix_kl",
    "target_token_kl": "kmtd_prefix_kl",
    "target_token_prior_kl": "kmtd_prefix_kl",
}


def as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def normalize_ft_loss_objective(value: Any) -> str:
    normalized = str(value or "standard_ft").strip().lower()
    if normalized not in FT_LOSS_OBJECTIVE_ALIASES:
        raise ValueError(
            f"Unsupported ft_loss_objective={value}. "
            f"Supported objectives: {', '.join(sorted(set(FT_LOSS_OBJECTIVE_ALIASES.values())))}"
        )
    return FT_LOSS_OBJECTIVE_ALIASES[normalized]


class WeightSwapContext:
    def __init__(
        self,
        weights_to_update: Dict[str, torch.nn.Parameter],
        ref_snapshot: Dict[str, torch.Tensor],
    ):
        self.weights_to_update = weights_to_update
        self.ref_snapshot = ref_snapshot
        self.current_snapshot: Dict[str, torch.Tensor] = {}

    def __enter__(self):
        with torch.no_grad():
            for name, param in self.weights_to_update.items():
                self.current_snapshot[name] = param.detach().clone()
                ref = self.ref_snapshot[name].to(device=param.device, dtype=param.dtype)
                param.copy_(ref)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with torch.no_grad():
            for name, param in self.weights_to_update.items():
                cur = self.current_snapshot[name].to(device=param.device, dtype=param.dtype)
                param.copy_(cur)
        return False


def build_full_batch_inputs(
    tok: AutoTokenizer,
    prompts: List[str],
    targets: List[str],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    prompt_inputs = tok(prompts, return_tensors="pt", padding=True).to(device)
    full_texts = [prompt + target for prompt, target in zip(prompts, targets)]
    full_inputs = tok(full_texts, return_tensors="pt", padding=True).to(device)

    prompt_token_counts = prompt_inputs["attention_mask"].sum(dim=1)
    full_token_counts = full_inputs["attention_mask"].sum(dim=1)
    full_seq_len = full_inputs["input_ids"].size(1)
    full_pad_counts = full_seq_len - full_token_counts
    prompt_end_positions = full_pad_counts + prompt_token_counts

    position_ids = torch.arange(full_seq_len, device=device).unsqueeze(0)
    label_mask = position_ids >= prompt_end_positions.unsqueeze(1)
    label_mask = label_mask & full_inputs["attention_mask"].bool()

    if bool((label_mask.sum(dim=1) == 0).any().item()):
        raise RuntimeError("A batch contains samples with zero target tokens after tokenization.")

    return full_inputs, label_mask


def compute_rewrite_reference_logits(
    model: AutoModelForCausalLM,
    prompt_inputs: Dict[str, torch.Tensor],
    weights_to_update: Dict[str, torch.nn.Parameter],
    ref_snapshot: Dict[str, torch.Tensor],
) -> torch.Tensor:
    with WeightSwapContext(weights_to_update, ref_snapshot):
        with torch.no_grad():
            ref_logits = model(**prompt_inputs).logits
    return ref_logits


def compute_target_token_prior_and_prefix_kl_loss(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    targets: List[str],
    device: torch.device,
    *,
    target_alpha: float,
    skip_above_target: bool,
    skip_prob: float,
    rewrite_kl_lambda: float,
    weights_to_update: Dict[str, torch.nn.Parameter],
    ref_snapshot: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    prompt_inputs = tok(prompts, return_tensors="pt", padding=True).to(device)
    full_inputs, label_mask = build_full_batch_inputs(tok, prompts, targets, device)
    target_mask = label_mask[:, 1:]

    ref_logits = None
    if rewrite_kl_lambda > 0.0:
        if ref_snapshot is None:
            raise ValueError("ref_snapshot must be provided when rewrite_kl_lambda > 0.")
        ref_logits = compute_rewrite_reference_logits(
            model=model,
            prompt_inputs=prompt_inputs,
            weights_to_update=weights_to_update,
            ref_snapshot=ref_snapshot,
        )

    logits = model(**full_inputs).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = full_inputs["input_ids"][:, 1:].contiguous()
    shift_log_p_theta = F.log_softmax(shift_logits, dim=-1)
    shift_p_theta = shift_log_p_theta.exp()

    alpha = float(target_alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"target_alpha must be in (0,1), got {alpha}")

    target_probs = shift_p_theta.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    if skip_above_target:
        active_mask = target_mask & (target_probs < float(skip_prob))
    else:
        active_mask = target_mask

    num_target_positions = int(target_mask.sum().item())
    num_active_positions = int(active_mask.sum().item())
    num_skipped_above_target_positions = (
        num_target_positions - num_active_positions if skip_above_target else 0
    )
    zero_loss = logits.sum() * 0.0
    avg_target_prob = target_probs[target_mask].mean().detach()

    if num_active_positions == 0:
        return {
            "loss": zero_loss,
            "kmtd_loss": zero_loss.detach(),
            "prefix_kl_loss": zero_loss.detach(),
            "avg_target_prob": avg_target_prob,
            "avg_target_prob_active": avg_target_prob,
            "avg_pstar_target": avg_target_prob,
            "num_target_positions": num_target_positions,
            "num_active_positions": num_active_positions,
            "num_skipped_above_target_positions": num_skipped_above_target_positions,
        }

    active_position_idx = torch.arange(num_active_positions, device=device)
    active_log_p_theta = shift_log_p_theta[active_mask]
    active_p_theta = shift_p_theta[active_mask]
    active_target_ids = shift_labels[active_mask]

    with torch.no_grad():
        p0 = active_p_theta.detach().clone()
        p_star = p0.clone()
        p0_target = p0[active_position_idx, active_target_ids].clamp(
            min=1e-12, max=1.0 - 1e-12
        )
        scale = ((1.0 - alpha) / (1.0 - p0_target).clamp_min(1e-6)).unsqueeze(1)
        p_star = p_star * scale
        p_star[active_position_idx, active_target_ids] = alpha
        p_star = p_star / p_star.sum(dim=-1, keepdim=True)

    kmtd_loss = F.kl_div(
        active_log_p_theta,
        p_star.log().clamp(min=-100),
        log_target=True,
        reduction="batchmean",
    )

    prefix_kl_loss = zero_loss
    if rewrite_kl_lambda > 0.0:
        prompt_logits = model(**prompt_inputs).logits
        prompt_attention_mask = prompt_inputs["attention_mask"]
        prefix_mask = prompt_attention_mask[:, :-1].bool() & prompt_attention_mask[:, 1:].bool()

        if bool(prefix_mask.any().item()):
            cur_log_probs = F.log_softmax(prompt_logits[:, :-1, :], dim=-1)
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            kl_per_pos = F.kl_div(
                cur_log_probs,
                ref_log_probs,
                log_target=True,
                reduction="none",
            ).sum(dim=-1)
            prefix_kl_loss = (kl_per_pos * prefix_mask).sum() / prefix_mask.sum()

    total_loss = kmtd_loss + float(rewrite_kl_lambda) * prefix_kl_loss

    avg_target_prob_active = target_probs[active_mask].mean().detach()
    avg_pstar_target = p_star[active_position_idx, active_target_ids].mean().detach()

    return {
        "loss": total_loss,
        "kmtd_loss": kmtd_loss.detach(),
        "prefix_kl_loss": prefix_kl_loss.detach(),
        "avg_target_prob": avg_target_prob,
        "avg_target_prob_active": avg_target_prob_active,
        "avg_pstar_target": avg_pstar_target,
        "num_target_positions": num_target_positions,
        "num_active_positions": num_active_positions,
        "num_skipped_above_target_positions": num_skipped_above_target_positions,
    }


def apply_ft_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_ft(model, tok, requests, hparams)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    device = torch.device(f'cuda:{hparams.device}')
    # model = model.to(device)
    # Update target and print info
    requests = deepcopy(requests)
    append_eos_to_target = as_bool(getattr(hparams, "append_eos_to_target", False))
    for request in requests:
        if request["target_new"] != " " and not request["target_new"].startswith(" "):
            # Space required for correct tokenization
            request["target_new"] = " " + request["target_new"]
        if (
            append_eos_to_target
            and tok.eos_token is not None
            and not request["target_new"].endswith(tok.eos_token)
        ):
            request["target_new"] += tok.eos_token
        print(
            f"Executing FT algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    
    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    ft_loss_objective = normalize_ft_loss_objective(
        getattr(hparams, "ft_loss_objective", "standard_ft")
    )
    target_alpha = float(getattr(hparams, "target_alpha", 0.1) or 0.1)
    pl_skip_above_target = as_bool(getattr(hparams, "pl_skip_above_target", False))
    skip_prob = float(getattr(hparams, "skip_prob", 1.0) or 1.0)
    rewrite_kl_lambda = float(getattr(hparams, "rewrite_kl_lambda", 0.0) or 0.0)
    ref_snapshot = (
        weights_copy if ft_loss_objective == "kmtd_prefix_kl" and rewrite_kl_lambda > 0.0 else None
    )

    print(f"FT loss objective: {ft_loss_objective}")
    if ft_loss_objective == "kmtd_prefix_kl":
        print(
            "KMTD config: "
            f"target_alpha={target_alpha}, "
            f"pl_skip_above_target={pl_skip_above_target}, "
            f"skip_prob={skip_prob}, "
            f"rewrite_kl_lambda={rewrite_kl_lambda}"
        )

    # Define inputs
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]
    
    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    # Update loop: intervene at layers simultaneously
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        for txt, tgt in zip(
            chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            opt.zero_grad()
            bs = len(txt)

            if ft_loss_objective == "kmtd_prefix_kl":
                if "t5" in hparams.model_name.lower() or "chatglm" in hparams.model_name.lower():
                    raise NotImplementedError(
                        "ft_loss_objective='kmtd_prefix_kl' is only implemented for causal LM FT."
                    )

                out = compute_target_token_prior_and_prefix_kl_loss(
                    model=model,
                    tok=tok,
                    prompts=txt,
                    targets=tgt,
                    device=device,
                    target_alpha=target_alpha,
                    skip_above_target=pl_skip_above_target,
                    skip_prob=skip_prob,
                    rewrite_kl_lambda=rewrite_kl_lambda,
                    weights_to_update=weights,
                    ref_snapshot=ref_snapshot,
                )

                if out["num_active_positions"] == 0:
                    print(
                        "Batch KMTD skipped: "
                        f"all {out['num_target_positions']} target positions already >= "
                        f"skip_prob={skip_prob:.6f}"
                    )
                    continue

                loss = out["loss"]
                print(
                    f"Batch KMTD loss {loss.item()} "
                    f"kmtd={float(out['kmtd_loss'])} "
                    f"prefix_kl={float(out['prefix_kl_loss'])} "
                    f"avg_target_prob={float(out['avg_target_prob'])} "
                    f"avg_target_prob_active={float(out['avg_target_prob_active'])} "
                    f"avg_pstar_target={float(out['avg_pstar_target'])} "
                    f"active_positions={out['num_active_positions']}/{out['num_target_positions']} "
                    f"skipped_above_target={out['num_skipped_above_target_positions']}"
                )
            else:
                inputs = tok(txt, return_tensors="pt", padding=True).to(device)
                target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(
                    device
                )
                if hparams.objective_optimization == 'prompt_last':
                    last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
                    if tok.unk_token_id is not None:
                        loss_mask = torch.ne(target_ids, tok.unk_token_id)
                    else:
                        loss_mask = torch.ones_like(target_ids, dtype=torch.bool)
                elif hparams.objective_optimization == 'target_new':
                    inputs_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
                    inputs_targets = tok(inputs_targets, return_tensors="pt", padding=True).to(device)
                    prompt_lens = inputs["attention_mask"].sum(dim=1)
                    full_lens = inputs_targets["attention_mask"].sum(dim=1)
                    target_lens = full_lens - prompt_lens
                    if bool((target_lens <= 0).any().item()):
                        raise ValueError("FT target_new objective received an empty target span.")
                    position_ids = inputs_targets["attention_mask"].long().cumsum(dim=1) - 1
                    label_mask = (
                        (position_ids >= prompt_lens.unsqueeze(1))
                        & inputs_targets["attention_mask"].bool()
                    )
                else:
                    print(f"{hparams.objective_optimization} has not been supported yet.")
                    raise NotImplementedError

                bs = inputs["input_ids"].shape[0]
                if 't5' in hparams.model_name.lower():
                    inputs['decoder_input_ids'] = target_ids
                    logits = model(**inputs).logits
                    unmasked_log_probs = logits.log_softmax(-1).gather(
                        -1,
                        inputs['decoder_input_ids'].unsqueeze(-1),
                    ).squeeze(-1)

                    mask = inputs['decoder_input_ids'] != -100
                    n_tokens = mask.float().sum()
                    avg_log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
                    nll = -avg_log_prob
                    loss = nll
                elif 'chatglm' in hparams.model_name.lower():
                    input_ids = inputs['input_ids'].tolist()
                    labels = target_ids.tolist()
                    assert len(input_ids) == len(labels)
                    len_batches = [
                        len(input_ids[i]) + len(labels[i]) + 1
                        for i in range(len(input_ids))
                    ]
                    len_max_batch = max(len_batches)
                    batch_input_ids = []
                    batch_labels = []
                    for x, y in zip(input_ids, labels):
                        len_padding = len_max_batch - len(x) - len(y)
                        if tok.padding_side and tok.padding_side == "left":
                            batch_label = [-100] * len_padding + [-100] * len(x) + y
                            batch_input_id = [0] * (len_padding) + x + y
                        else:
                            batch_label = [-100] * len(x) + y + [-100] * len_padding
                            batch_input_id = x + y + [0] * (len_padding)

                        tensor_input_ids = torch.tensor(batch_input_id, dtype=torch.long)
                        tensor_labels = torch.tensor(batch_label, dtype=torch.long)
                        batch_input_ids.append(tensor_input_ids)
                        batch_labels.append(tensor_labels)

                    batch_input_ids = torch.stack(batch_input_ids).to(device)
                    batch_labels = torch.stack(batch_labels).to(device)
                    lm_logits = model(input_ids=batch_input_ids)['logits']
                    lm_logits = lm_logits.to(torch.float32)
                    shift_logits = lm_logits[..., :-1, :].contiguous()
                    shift_labels = batch_labels[..., 1:].contiguous()
                    loss_fct = CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    loss = loss.to(lm_logits.dtype)
                else:
                    if hparams.objective_optimization == 'prompt_last':
                        probs = torch.nn.functional.log_softmax(
                            model(**inputs).logits[torch.arange(bs), last_token_inds],
                            dim=-1,
                        )
                        loss = -(torch.gather(probs, 1, target_ids) * loss_mask).sum(
                            1
                        ) / loss_mask.sum(1)
                        loss = loss.mean()
                    elif hparams.objective_optimization == 'target_new':
                        logits = model(**inputs_targets).logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = inputs_targets['input_ids'][..., 1:].contiguous()
                        loss_fct = CrossEntropyLoss(reduction='none')
                        loss = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                        )
                        loss = loss.view(bs, -1)
                        loss = (loss * label_mask[:, 1:]).sum(1) / label_mask[:, 1:].sum(1)
                        loss = loss.mean()
                    else:
                        raise NotImplementedError
                print(f"Batch loss {loss.item()}")

            loss_meter.update(loss.item(), n=bs)

            if loss.item() >= 1e-2:
                loss.backward()
                opt.step()

            if type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )

        print(f"Total loss {loss_meter.avg}")

        if loss_meter.avg < 1e-2:
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
