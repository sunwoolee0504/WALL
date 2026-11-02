"""Shared WALL attack primitives for collection and imagined counterfactuals.

Each selected agent's adversarial action is computed against the original joint
policy action (Definition 4.1). QPLEX needs its full action-dependent joint Q;
VDN and QMIX permit the cheaper local argmin by monotonicity.
"""
import torch as th
import torch.nn.functional as F


def _legal_actions(avail_actions):
    """Use a finite no-op distribution for entirely padded/dead rows."""
    legal = avail_actions.bool().clone()
    legal[..., 0] |= ~legal.any(dim=-1)
    return legal


def joint_q(mixer, args, all_q, state, actions, avail_actions, detach_advantage=False):
    """Full Q_tot at a joint action, preserving derivatives with respect to Q_i."""
    if actions.ndim == 3:
        actions = actions.squeeze(-1)
    selected = all_q.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
    if mixer is None:
        return selected.sum(dim=-1)
    selected = selected.unsqueeze(1)
    state = state.reshape(state.shape[0], 1, -1)
    if getattr(args, "mixer", "") in ("dmaq", "qplex"):
        max_q = all_q.masked_fill(~_legal_actions(avail_actions), -th.inf).amax(-1).unsqueeze(1)
        onehot = F.one_hot(actions.long(), all_q.shape[-1]).to(all_q.dtype).unsqueeze(1)
        value = mixer(selected, state, is_v=True)
        advantage = mixer(selected, state, actions=onehot, max_q_i=max_q,
                          is_v=False, detach_advantage=detach_advantage)
        return (value + advantage).reshape(all_q.shape[0])
    return mixer(selected, state).reshape(all_q.shape[0])


@th.no_grad()
def attack_actions(mixer, args, all_q, state, actions, avail_actions, target_mask):
    """Return simultaneous unilateral minimizers; leave unselected agents alone."""
    if actions.ndim == 3:
        actions = actions.squeeze(-1)
    actions = actions.long()
    legal = _legal_actions(avail_actions)
    targets = target_mask.bool() & (avail_actions.bool().sum(-1) > 1)
    if getattr(args, "mixer", "") not in ("dmaq", "qplex"):
        worst = all_q.masked_fill(~legal, th.inf).argmin(-1)
        return th.where(targets, worst, actions)

    positions = targets.nonzero(as_tuple=False)
    if positions.numel() == 0:
        return actions.clone()
    batch_ids, agent_ids = positions.unbind(-1)
    n_candidates, n_actions = positions.shape[0], all_q.shape[-1]
    candidate_actions = actions[batch_ids, None, :].expand(-1, n_actions, -1).clone()
    choices = th.arange(n_actions, device=actions.device)
    candidate_actions[th.arange(n_candidates, device=actions.device)[:, None],
                      choices[None, :], agent_ids[:, None]] = choices[None, :]
    candidate_q = all_q[batch_ids, None].expand(-1, n_actions, -1, -1)
    candidate_state = state[batch_ids, None].expand(-1, n_actions, *state.shape[1:])
    candidate_avail = avail_actions[batch_ids, None].expand(-1, n_actions, -1, -1)
    scores = joint_q(mixer, args, candidate_q.reshape(-1, *all_q.shape[1:]),
                     candidate_state.reshape(n_candidates * n_actions, -1),
                     candidate_actions.reshape(-1, actions.shape[-1]),
                     candidate_avail.reshape(-1, *avail_actions.shape[1:]))
    scores = scores.reshape(n_candidates, n_actions)
    scores = scores.masked_fill(~legal[batch_ids, agent_ids], th.inf)
    attacked = actions.clone()
    attacked[batch_ids, agent_ids] = scores.argmin(-1)
    return attacked


def select_followup(mixer, args, all_q, state, actions, attacked_actions,
                    avail_actions, initial_agents, obs=None):
    """Eq. (1)-(2) Q-output update and masked KL ranking.

    The normal Q_tot is a fixed baseline. The virtual step acts directly on the
    Q outputs, not shared policy parameters. No parameter or optimizer is copied
    or mutated. Returns [B,m] indices, padded with -1 when agents are ineligible.
    """
    count = min(int(args.num_followup_agents), all_q.shape[1] - 1)
    if count <= 0:
        return th.empty(all_q.shape[0], 0, dtype=th.long, device=all_q.device)
    method = getattr(args, "followup_selection", getattr(args, "followup_method", "q_gradient"))
    if getattr(args, "followup_l2", False):
        method = "l2"
    initial_agents = initial_agents.reshape(-1).long()
    rows = th.arange(all_q.shape[0], device=all_q.device)
    eligible = avail_actions.bool().sum(-1) > 1
    eligible = eligible.clone()
    eligible[rows, initial_agents] = False
    if method == "l2":
        if obs is None:
            raise ValueError("L2 follow-up selection requires observations")
        scores = -(obs - obs[rows, initial_agents].unsqueeze(1)).square().sum(-1)
    elif method == "q_gradient":
        with th.enable_grad():
            q = all_q.detach().float().requires_grad_(True)
            # The detached normal-Q baseline has zero derivative. Omitting
            # its forward pass preserves the exact input-Q gradient.
            attacked = joint_q(mixer, args, q, state.float(), attacked_actions, avail_actions)
            gradient, = th.autograd.grad(-attacked.sum(), q, create_graph=False)
        rate = float(getattr(args, "followup_q_lr", getattr(args, "lr", 0.0005)))
        updated = q.detach() - rate * gradient.detach()
        legal = _legal_actions(avail_actions)
        # A finite sentinel keeps 0 * (log(0) - log(0)) out of padded actions.
        log_before = F.log_softmax(q.detach().double().masked_fill(~legal, -1e9), dim=-1)
        log_after = F.log_softmax(updated.double().masked_fill(~legal, -1e9), dim=-1)
        # Float64 for this small reduction prevents cancellation from making
        # tiny, but meaningful, KL differences negative at the default lr.
        p = log_before.double().exp()
        scores = (p * (log_before.double() - log_after.double())).sum(-1).clamp_min(0)
    else:
        raise ValueError("Unsupported followup_selection %r; use q_gradient or l2" % method)
    scores = scores.masked_fill(~eligible, -th.inf)
    chosen = scores.argsort(dim=-1, descending=True, stable=True)[:, :count]
    return chosen.masked_fill(~eligible.gather(1, chosen), -1)
