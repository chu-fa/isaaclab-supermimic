"""Independent tensor definitions of paper Equations 2 and 3."""

import torch


def imagined_objective(rewards, terminations, terminal_value, discount):
    """Inputs [time, batch]; reward at a terminal transition is retained."""
    alive_before = torch.cat(
        (torch.ones_like(terminations[:1]), (1 - terminations).cumprod(dim=0)), dim=0
    )
    discounts = torch.as_tensor(
        discount, dtype=rewards.dtype, device=rewards.device
    ).pow(torch.arange(rewards.shape[0] + 1, device=rewards.device))
    reward_return = (discounts[:-1, None] * alive_before[:-1] * rewards).sum(dim=0)
    objective = reward_return + discounts[-1] * alive_before[-1] * terminal_value
    return objective, alive_before[-1]


@torch.no_grad()
def lambda_returns(rewards, terminations, values, discount, trace_decay):
    """Fixed critic targets; values contains the additional horizon bootstrap."""
    result = torch.empty_like(rewards)
    continuation = values[-1]
    for step in reversed(range(rewards.shape[0])):
        continuation = rewards[step] + discount * (1 - terminations[step]) * (
            (1 - trace_decay) * values[step + 1] + trace_decay * continuation
        )
        result[step] = continuation
    return result
