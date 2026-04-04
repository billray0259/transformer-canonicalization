import torch
import math
from torch.func import functional_call

def masked_token_pseudo_perplexity(model, tokenizer, texts, overrides=None):
    model_device = next(model.parameters()).device
    encoded = tokenizer(texts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(model_device)
    attention_mask = encoded["attention_mask"].to(model_device)
    special_token_ids = set(tokenizer.all_special_ids)
    losses = []

    if overrides is not None:
        overrides = {
            name: value.to(model_device) if value.device != model_device else value
            for name, value in overrides.items()
        }

    with torch.no_grad():
        for batch_index in range(input_ids.shape[0]):
            for token_index in range(input_ids.shape[1]):
                if attention_mask[batch_index, token_index].item() == 0:
                    continue

                token_id = input_ids[batch_index, token_index].item()
                if token_id in special_token_ids:
                    continue

                masked_input_ids = input_ids[batch_index:batch_index + 1].clone()
                masked_attention_mask = attention_mask[batch_index:batch_index + 1]
                labels = torch.full_like(masked_input_ids, -100)
                labels[0, token_index] = token_id
                masked_input_ids[0, token_index] = tokenizer.mask_token_id

                if overrides is None:
                    outputs = model(
                        input_ids=masked_input_ids,
                        attention_mask=masked_attention_mask,
                        labels=labels,
                    )
                else:
                    outputs = functional_call(
                        model,
                        overrides,
                        (),
                        {
                            "input_ids": masked_input_ids,
                            "attention_mask": masked_attention_mask,
                            "labels": labels,
                        },
                    )

                losses.append(outputs.loss.item())

    assert losses, "Expected at least one non-special token to evaluate."
    return math.exp(sum(losses) / len(losses))


def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """Approximate a doubly stochastic matrix with Sinkhorn iterations."""
    log_transport = log_alpha
    for _ in range(n_iters):
        log_transport = log_transport - torch.logsumexp(
            log_transport, dim=-1, keepdim=True
        )
        log_transport = log_transport - torch.logsumexp(
            log_transport, dim=-2, keepdim=True
        )
    return log_transport.exp()