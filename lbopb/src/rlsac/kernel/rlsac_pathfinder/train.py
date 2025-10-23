# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng

from __future__ import annotations

import json
import time as _pytime
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from .env_domain import DomainPathfinderEnv, Goal
from .domain import get_domain_spec
from lbopb.src.rlsac.application.rlsac_nsclc.models import DiscretePolicy, QNetwork
from lbopb.src.rlsac.application.rlsac_nsclc.replay_buffer import ReplayBuffer
from lbopb.src.rlsac.application.rlsac_nsclc.utils import soft_update, select_device_from_config, discrete_entropy


def _load_config(cfg_path: Path) -> Dict[str, Any]:
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def build_env_from_config(cfg_path: Path) -> DomainPathfinderEnv:
    cfg = _load_config(cfg_path)
    domain = str(cfg.get("domain", "pem")).lower()
    spec = get_domain_spec(domain)
    s0_cfg = cfg.get("initial_state", {"b": 3.0, "n_comp": 3, "perim": 5.0, "fidelity": 0.4})
    st_cfg = cfg.get("target_state", {"b": 0.5, "n_comp": 1, "perim": 2.0, "fidelity": 0.9})
    tol_cfg = cfg.get("tolerance", {"b": 0.1, "n": 0.5, "p": 0.2, "f": 0.05})
    init_state = spec.state_cls(
        b=float(s0_cfg.get("b", 3.0)),
        n_comp=int(s0_cfg.get("n_comp", 3)),
        perim=float(s0_cfg.get("perim", 5.0)),
        fidelity=float(s0_cfg.get("fidelity", 0.4)),
    )
    goal = Goal(
        target=spec.state_cls(
            b=float(st_cfg.get("b", 0.5)),
            n_comp=int(st_cfg.get("n_comp", 1)),
            perim=float(st_cfg.get("perim", 2.0)),
            fidelity=float(st_cfg.get("fidelity", 0.9)),
        ),
        tol_b=float(tol_cfg.get("b", 0.1)),
        tol_n=float(tol_cfg.get("n", 0.5)),
        tol_p=float(tol_cfg.get("p", 0.2)),
        tol_f=float(tol_cfg.get("f", 0.05)),
    )
    max_steps = int(cfg.get("episode_max_steps", 64))
    improve_w = float(cfg.get("improve_weight", 1.0))
    step_penalty = float(cfg.get("step_penalty", 0.01))
    include_identity = bool(cfg.get("include_identity", False))
    env = DomainPathfinderEnv(
        spec,
        init_state=init_state,
        goal=goal,
        max_steps=max_steps,
        improve_weight=improve_w,
        step_penalty=step_penalty,
        include_identity=include_identity,
    )
    return env


def train(config_path: str | Path | None = None) -> Path:
    mod_dir = Path(__file__).resolve().parent
    cfg_path = Path(config_path) if config_path else (mod_dir / "config.json")
    cfg = _load_config(cfg_path)
    device = select_device_from_config(cfg_path)

    debug = bool(cfg.get("debug", False))
    log_every_step = bool(cfg.get("log_every_step", True))
    log_to_file = bool(cfg.get("log_to_file", True))

    class RunLogger:
        def __init__(self, path: Path, append: bool = False):
            self.path = path
            self.f = open(path, 'a' if append else 'w', encoding='utf-8', newline='')
        def write_line(self, text: str) -> None:
            try:
                self.f.write(text + "\r\n"); self.f.flush()
            except Exception:
                pass
        def close(self) -> None:
            try:
                self.f.flush(); self.f.close()
            except Exception:
                pass

    def dbg(msg: str) -> None:
        if debug:
            print(f"[DEBUG] {msg}")
            if logger:
                logger.write_line(f"[DEBUG] {msg}")

    def step_log(msg: str) -> None:
        if log_every_step or debug:
            print(msg)
            if logger:
                logger.write_line(msg)

    # 环境
    env = build_env_from_config(cfg_path)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.high - env.action_space.low
    dbg(f"环境初始化: obs_dim={obs_dim}, n_actions={n_actions}")

    lr_actor = float(cfg.get("learning_rate_actor", 3e-4))
    lr_critic = float(cfg.get("learning_rate_critic", 3e-4))
    gamma = float(cfg.get("gamma", 0.99))
    tau = float(cfg.get("tau", 0.005))
    buffer_size = int(cfg.get("buffer_size", 50000))
    batch_size = int(cfg.get("batch_size", 64))
    target_entropy = float(cfg.get("target_entropy", -1.0 * n_actions))
    learn_alpha = bool(cfg.get("learn_alpha", True))
    init_alpha = float(cfg.get("alpha", 0.2))
    total_steps = int(cfg.get("total_steps", 5000))
    start_steps = int(cfg.get("start_steps", 1000))
    update_after = int(cfg.get("update_after", 1000))
    update_every = int(cfg.get("update_every", 50))
    updates_per_step = int(cfg.get("updates_per_step", 1))
    minibatch_floor = int(cfg.get("minibatch_floor", 1))

    policy = DiscretePolicy(obs_dim, n_actions).to(device)
    q1 = QNetwork(obs_dim, n_actions).to(device)
    q2 = QNetwork(obs_dim, n_actions).to(device)
    tgt_q1 = QNetwork(obs_dim, n_actions).to(device)
    tgt_q2 = QNetwork(obs_dim, n_actions).to(device)
    tgt_q1.load_state_dict(q1.state_dict()); tgt_q2.load_state_dict(q2.state_dict())
    opt_pi = torch.optim.Adam(policy.parameters(), lr=lr_actor)
    opt_q1 = torch.optim.Adam(q1.parameters(), lr=lr_critic)
    opt_q2 = torch.optim.Adam(q2.parameters(), lr=lr_critic)

    log_alpha = torch.tensor(float(torch.log(torch.tensor(init_alpha))), requires_grad=learn_alpha, device=device)
    opt_alpha = torch.optim.Adam([log_alpha], lr=lr_actor) if learn_alpha else None

    buf = ReplayBuffer(buffer_size, obs_dim)
    state = env.reset()
    last_q1 = last_q2 = last_pi = None
    last_alpha = float(log_alpha.exp().item()) if learn_alpha else init_alpha
    logger: RunLogger | None = None

    # 运行目录与日志
    repo_root = Path(__file__).resolve().parents[3]
    base_out = repo_root / Path(cfg.get("output_dir", "out_pathfinder"))
    base_out.mkdir(parents=True, exist_ok=True)
    run_dir = base_out / ("train_" + str(int(_pytime.time())))
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    if log_to_file:
        logger = RunLogger(log_path, append=False)
        logger.write_line(f"# TRAIN START { _pytime.strftime('%Y-%m-%d %H:%M:%S', _pytime.localtime()) }")
        logger.write_line(f"device={device}")
        logger.write_line(f"config={json.dumps(cfg, ensure_ascii=False)}")

    # 导出 op 索引
    try:
        if hasattr(env, "op2idx"):
            mapping = getattr(env, "op2idx")
            text = json.dumps(mapping, ensure_ascii=False, indent=2)
            text = text.replace("\r\n", "\n").replace("\n", "\r\n")
            (run_dir / "op_index.json").write_text(text, encoding="utf-8")
    except Exception:
        pass

    for t in range(1, total_steps + 1):
        with torch.no_grad():
            if t < start_steps:
                action = env.action_space.sample().item()
            else:
                s = state.to(device).unsqueeze(0)
                logits, probs = policy(s)
                dist = torch.distributions.Categorical(probs=probs)
                action = int(dist.sample().item())

        s2, r, d, info = env.step(torch.tensor([action], dtype=torch.int32))
        buf.push(state, action, r, s2, d)
        state = s2

        updates_done = 0
        if t >= update_after and t % update_every == 0 and len(buf) >= max(1, minibatch_floor):
            for upd in range(max(1, updates_per_step)):
                s_b, a_b, r_b, s2_b, d_b = buf.sample(batch_size)
                s_b = s_b.to(device); a_b = a_b.to(device); r_b = r_b.to(device)
                s2_b = s2_b.to(device); d_b = d_b.to(device)

                with torch.no_grad():
                    logits2, probs2 = policy(s2_b)
                    q1_t = tgt_q1(s2_b); q2_t = tgt_q2(s2_b)
                    q_min = torch.min(q1_t, q2_t)
                    logp2 = torch.log(torch.clamp(probs2, 1e-8, 1.0))
                    alpha = log_alpha.exp().detach() if learn_alpha else torch.tensor(init_alpha, device=device)
                    v_s2 = (probs2 * (q_min - alpha * logp2)).sum(dim=-1)
                    y = r_b + gamma * (1.0 - d_b) * v_s2

                q1_sa = q1(s_b).gather(1, a_b.view(-1, 1)).squeeze(1)
                q2_sa = q2(s_b).gather(1, a_b.view(-1, 1)).squeeze(1)
                loss_q1 = F.mse_loss(q1_sa, y)
                loss_q2 = F.mse_loss(q2_sa, y)
                opt_q1.zero_grad(); loss_q1.backward(); opt_q1.step()
                opt_q2.zero_grad(); loss_q2.backward(); opt_q2.step()
                last_q1 = float(loss_q1.item()); last_q2 = float(loss_q2.item())

                logits, probs = policy(s_b)
                logp = torch.log(torch.clamp(probs, 1e-8, 1.0))
                q1_pi = q1(s_b); q2_pi = q2(s_b)
                q_pi = torch.min(q1_pi, q2_pi)
                alpha = log_alpha.exp() if learn_alpha else torch.tensor(init_alpha, device=device)
                loss_pi = (probs * (alpha * logp - q_pi)).sum(dim=-1).mean()
                opt_pi.zero_grad(); loss_pi.backward(); opt_pi.step()
                last_pi = float(loss_pi.item())

                if learn_alpha and opt_alpha is not None:
                    ent = discrete_entropy(probs).detach()
                    loss_alpha = -(log_alpha * (target_entropy - ent).detach()).mean()
                    opt_alpha.zero_grad(); loss_alpha.backward(); opt_alpha.step()

                soft_update(tgt_q1, q1, tau); soft_update(tgt_q2, q2, tau)
            updates_done = max(1, updates_per_step)

        step_log(
            "[STEP {t}] a={a} r={r:.6f} d={d} buf={buf} upd={upd} "
            "loss_q1={lq1} loss_q2={lq2} loss_pi={lpi} alpha={alpha} dist={dist}".format(
                t=t, a=action, r=r, d=d, buf=len(buf), upd=updates_done,
                lq1=("{:.6f}".format(last_q1) if last_q1 is not None else "-"),
                lq2=("{:.6f}".format(last_q2) if last_q2 is not None else "-"),
                lpi=("{:.6f}".format(last_pi) if last_pi is not None else "-"),
                alpha=("{:.6f}".format(float(log_alpha.exp().item())) if learn_alpha else f"{init_alpha:.6f}"),
                dist=float(info.get("dist", 0.0)),
            )
        )

        if d:
            state = env.reset()

    # 保存权重
    torch.save(policy.state_dict(), run_dir / "policy.pt")
    torch.save(q1.state_dict(), run_dir / "q1.pt")
    torch.save(q2.state_dict(), run_dir / "q2.pt")
    print(f"Training finished. Artifacts at: {run_dir}")
    return run_dir


def extract_operator_package(run_dir: str | Path, config_path: str | Path | None = None, max_len: int | None = None) -> Dict[str, Any]:
    """贪心解码提取算子包，写入对应域的 operator_packages.json。"""
    mod_dir = Path(__file__).resolve().parent
    cfg_path = Path(config_path) if config_path else (mod_dir / "config.json")
    cfg = _load_config(cfg_path)
    domain = str(cfg.get("domain", "pem")).lower()
    env = build_env_from_config(cfg_path)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.high - env.action_space.low

    # 加载策略
    run_dir = Path(run_dir)
    policy = DiscretePolicy(obs_dim, n_actions)
    policy.load_state_dict(torch.load(run_dir / "policy.pt", map_location="cpu"))
    policy.eval()

    # 贪心 rollout
    names = {v: k for k, v in env.op2idx.items()}
    seq: List[str] = []
    s = env.reset().unsqueeze(0)
    steps = 0
    max_len = int(max_len or env._max_steps)
    with torch.no_grad():
        while steps < max_len:
            logits, probs = policy(s)
            a = int(torch.argmax(probs, dim=-1).item())
            seq.append(names.get(a, str(a)))
            s2, r, d, _ = env.step(a)
            s = s2.unsqueeze(0)
            steps += 1
            if d:
                break

    # 写入 package 字典
    pkg = {
        "id": f"pkg_{domain}_{int(_pytime.time())}",
        "domain": domain,
        "initial": {
            "b": env._init_state_cfg.b,
            "n_comp": env._init_state_cfg.n_comp,
            "perim": env._init_state_cfg.perim,
            "fidelity": env._init_state_cfg.fidelity,
        },
        "target": {
            "b": env._goal.target.b,
            "n_comp": env._goal.target.n_comp,
            "perim": env._goal.target.perim,
            "fidelity": env._goal.target.fidelity,
        },
        "sequence": seq,
        "max_steps": max_len,
        "created_at": int(_pytime.time()),
    }

    dict_path = mod_dir / get_domain_spec(domain).dict_filename
    arr: List[Dict[str, Any]] = []
    if dict_path.exists():
        try:
            arr = json.loads(dict_path.read_text(encoding="utf-8"))
        except Exception:
            arr = []
    arr.append(pkg)
    text = json.dumps(arr, ensure_ascii=False, indent=2)
    text = text.replace("\r\n", "\n").replace("\n", "\r\n")
    dict_path.write_text(text, encoding="utf-8")
    return pkg


if __name__ == "__main__":
    out = train()
    print(out)





