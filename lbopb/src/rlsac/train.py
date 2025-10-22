# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
import os
import sys

# Support both package and script execution
try:
    from .env import DummyEnv
    from .models import DiscretePolicy, QNetwork
    from .replay_buffer import ReplayBuffer
    from .utils import soft_update, select_device_from_config, discrete_entropy
except ImportError:  # running as a script (no package parent)
    import sys as _sys
    from pathlib import Path as _Path
    _PKG_PARENT = _Path(__file__).resolve().parents[1]  # lbopb/src
    if str(_PKG_PARENT) not in _sys.path:
        _sys.path.insert(0, str(_PKG_PARENT))
    from rlsac.env import DummyEnv  # type: ignore
    from rlsac.models import DiscretePolicy, QNetwork  # type: ignore
    from rlsac.replay_buffer import ReplayBuffer  # type: ignore
    from rlsac.utils import soft_update, select_device_from_config, discrete_entropy  # type: ignore


def _load_config(cfg_path: Path) -> Dict[str, Any]:
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _prefill_from_json(buf: ReplayBuffer, json_path: Path, obs_dim: int) -> int:
    if not json_path.exists():
        return 0
    data = json.loads(json_path.read_text(encoding="utf-8"))
    # If the file is a case-packages JSON (e.g., copied from operator_crosswalk.json),
    # we do not attempt to convert to RL tuples here.
    if isinstance(data, dict) and "case_packages" in data:
        return 0
    cnt = 0
    # Expected schema (minimal): [{"state":[...], "action":int, "reward":float, "next_state":[...], "done":bool}, ...]
    for item in data:
        try:
            s = torch.tensor(item["state"], dtype=torch.float32)
            a = int(item["action"])  # discrete index
            r = float(item.get("reward", 0.0))
            s2 = torch.tensor(item.get("next_state", item["state"]), dtype=torch.float32)
            d = bool(item.get("done", False))
            if s.numel() == obs_dim and s2.numel() == obs_dim:
                buf.push(s, a, r, s2, d)
                cnt += 1
        except Exception:
            continue
    return cnt


def _select_run_dir(base_out: Path, select: str | None = None) -> Path | None:
    """Scan base_out/train_* and prompt to resume one.

    Returns selected run dir Path or None for a new run.
    """
    if not base_out.exists():
        return None
    runs = sorted([p for p in base_out.glob("train_*") if p.is_dir()])
    if not runs:
        return None
    # Auto selection by argument/env
    if select:
        s = str(select).strip().lower()
        if s == "new":
            return None
        if s == "latest":
            return runs[-1]
        if s.isdigit():
            idx = int(s)
            if 0 <= idx < len(runs):
                return runs[idx]
    print("Detected previous training runs:")
    for i, r in enumerate(runs):
        print(f"  [{i}] {r.name}")
    choice = input("Enter index to resume, or press Enter for new: ").strip()
    if choice.isdigit():
        idx = int(choice)
        if 0 <= idx < len(runs):
            return runs[idx]
    return None

def train(
    config_path: str | Path | None = None,
    data_json: str | Path | None = None,
) -> None:
    """Train discrete SAC on DummyEnv with optional prefill from JSON.

    - Default device CPU; use config.json to enable GPU (if available).
    - Config schema includes learning rate, gamma, tau, batch_size, buffer_size, alpha, target_entropy.
    - operator_crosswalk_train.json is used for optional imitation prefill.
    """

    mod_dir = Path(__file__).resolve().parent
    cfg_path = Path(config_path) if config_path else (mod_dir / "config.json")
    cfg = _load_config(cfg_path)
    device = select_device_from_config(cfg_path)

    # 调试模式：环境变量 RLSAC_DEBUG 优先，其次读取配置项 debug
    debug = bool(os.environ.get("RLSAC_DEBUG", "") or cfg.get("debug", False))

    def dbg(msg: str) -> None:
        if debug:
            print(f"[DEBUG] {msg}")

    dbg(f"配置文件: {cfg_path}")
    dbg(f"设备选择: {device}")

    env = DummyEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.high - env.action_space.low
    dbg(f"环境初始化: obs_dim={obs_dim}, n_actions={n_actions}")

    # Hyperparameters
    lr_actor = float(cfg.get("learning_rate_actor", 3e-4))
    lr_critic = float(cfg.get("learning_rate_critic", 3e-4))
    gamma = float(cfg.get("gamma", 0.99))
    tau = float(cfg.get("tau", 0.005))
    buffer_size = int(cfg.get("buffer_size", 100000))
    batch_size = int(cfg.get("batch_size", 64))
    target_entropy = float(cfg.get("target_entropy", -1.0 * n_actions))
    learn_alpha = bool(cfg.get("learn_alpha", True))
    init_alpha = float(cfg.get("alpha", 0.2))
    total_steps = int(cfg.get("total_steps", 2000))
    start_steps = int(cfg.get("start_steps", 1000))
    update_after = int(cfg.get("update_after", 1000))
    update_every = int(cfg.get("update_every", 50))
    updates_per_step = int(cfg.get("updates_per_step", 1))
    minibatch_floor = int(cfg.get("minibatch_floor", 1))  # allow training with very small buffers

    # Output directory at project root (repo_root/out by default) and resume selection
    # repo_root ~= rlsac/../../..
    try:
        repo_root = mod_dir.parents[2]
    except Exception:
        repo_root = Path.cwd()
    out_cfg = cfg.get("output_dir", "out")
    out_path = Path(out_cfg)
    base_out = out_path if out_path.is_absolute() else (repo_root / out_path)
    base_out.mkdir(parents=True, exist_ok=True)
    # parse CLI/env selection to avoid interactive block
    select_arg: str | None = None
    args = sys.argv[1:]
    if "--new" in args:
        select_arg = "new"
    elif "--latest" in args:
        select_arg = "latest"
    elif "--resume" in args:
        try:
            idx_pos = args.index("--resume") + 1
            select_arg = args[idx_pos]
        except Exception:
            select_arg = None
    if not select_arg:
        env_sel = os.environ.get("RLSAC_SELECT", "").strip()
        if env_sel:
            select_arg = env_sel
        elif cfg.get("select"):
            select_arg = str(cfg.get("select")).strip()
    resume_dir = _select_run_dir(base_out, select=select_arg)
    dbg(f"选择逻辑: args={args}, RLSAC_SELECT={os.environ.get('RLSAC_SELECT','')!r}, cfg.select={cfg.get('select', None)!r}, 自动选择={select_arg!r}")
    import time as _pytime
    if resume_dir is None:
        run_dir = base_out / ("train_" + str(int(_pytime.time())))
        run_dir.mkdir(parents=True, exist_ok=True)
        resume = False
    else:
        run_dir = resume_dir
        resume = True
    dbg(f"训练运行目录: {run_dir} | 模式: {'续跑' if resume else '新建'}")

    # Networks
    policy = DiscretePolicy(obs_dim, n_actions).to(device)
    q1 = QNetwork(obs_dim, n_actions).to(device)
    q2 = QNetwork(obs_dim, n_actions).to(device)
    tgt_q1 = QNetwork(obs_dim, n_actions).to(device)
    tgt_q2 = QNetwork(obs_dim, n_actions).to(device)
    # Resume weights if requested
    if resume:
        try:
            p_p = run_dir / "policy.pt"
            q1_p = run_dir / "q1.pt"
            q2_p = run_dir / "q2.pt"
            if p_p.exists():
                policy.load_state_dict(torch.load(p_p, map_location=device))
            if q1_p.exists():
                q1.load_state_dict(torch.load(q1_p, map_location=device))
            if q2_p.exists():
                q2.load_state_dict(torch.load(q2_p, map_location=device))
            print(f"Resumed from {run_dir}")
        except Exception as e:
            print(f"Warning: failed to resume weights: {e}")
    tgt_q1.load_state_dict(q1.state_dict())
    tgt_q2.load_state_dict(q2.state_dict())

    opt_pi = torch.optim.Adam(policy.parameters(), lr=lr_actor)
    opt_q1 = torch.optim.Adam(q1.parameters(), lr=lr_critic)
    opt_q2 = torch.optim.Adam(q2.parameters(), lr=lr_critic)

    # Temperature parameter alpha (either fixed or learnable)
    log_alpha = torch.tensor(float(torch.log(torch.tensor(init_alpha))), requires_grad=learn_alpha, device=device)
    opt_alpha = torch.optim.Adam([log_alpha], lr=lr_actor) if learn_alpha else None

    # Replay buffer
    buf = ReplayBuffer(buffer_size, obs_dim)
    data_path = Path(data_json) if data_json else (mod_dir / "operator_crosswalk_train.json")
    prefilled = _prefill_from_json(buf, data_path, obs_dim)
    dbg(f"预填充回放池: 来自 {data_path}, 条目数={prefilled}")

    state = env.reset()
    ep_ret = 0.0
    for t in range(1, total_steps + 1):
        # Select action
        with torch.no_grad():
            if t < start_steps:
                action = env.action_space.sample().item()
                dbg(f"t={t} 随机动作: a={action}")
            else:
                s = state.to(device).unsqueeze(0)
                logits, probs = policy(s)
                dist = torch.distributions.Categorical(probs=probs)
                action = int(dist.sample().item())
                if debug:
                    dbg(f"t={t} 策略动作: a={action} | probs={probs.squeeze(0).detach().cpu().numpy()}")

        s2, r, d, _ = env.step(torch.tensor([action], dtype=torch.int32))
        buf.push(state, action, r, s2, d)
        state = s2
        ep_ret += r
        if debug:
            dbg(f"t={t} 交互: r={r:.6f}, done={d}, 回放池大小={len(buf)}")

        # Update
        if t >= update_after and t % update_every == 0 and len(buf) >= max(1, minibatch_floor):
            # Perform multiple gradient updates per environment step
            for upd in range(max(1, updates_per_step)):
                s_b, a_b, r_b, s2_b, d_b = buf.sample(batch_size)
                s_b = s_b.to(device)
                a_b = a_b.to(device)
                r_b = r_b.to(device)
                s2_b = s2_b.to(device)
                d_b = d_b.to(device)

                # Target V(s') using target critics + current policy
                with torch.no_grad():
                    logits2, probs2 = policy(s2_b)
                    q1_t = tgt_q1(s2_b)
                    q2_t = tgt_q2(s2_b)
                    q_min = torch.min(q1_t, q2_t)
                    logp2 = torch.log(torch.clamp(probs2, 1e-8, 1.0))
                    alpha = log_alpha.exp().detach() if learn_alpha else torch.tensor(init_alpha, device=device)
                    v_s2 = (probs2 * (q_min - alpha * logp2)).sum(dim=-1)
                    y = r_b + gamma * (1.0 - d_b) * v_s2

                # Critic loss: MSE(Q(s,a), y)
                q1_sa = q1(s_b).gather(1, a_b.view(-1, 1)).squeeze(1)
                q2_sa = q2(s_b).gather(1, a_b.view(-1, 1)).squeeze(1)
                loss_q1 = F.mse_loss(q1_sa, y)
                loss_q2 = F.mse_loss(q2_sa, y)
                opt_q1.zero_grad(); loss_q1.backward(); opt_q1.step()
                opt_q2.zero_grad(); loss_q2.backward(); opt_q2.step()
                if debug:
                    dbg(f"t={t} upd={upd+1} Critic: loss_q1={loss_q1.item():.6f}, loss_q2={loss_q2.item():.6f}")

                # Actor loss: E[alpha * log pi(a|s) - Q(s,a)]
                logits, probs = policy(s_b)
                logp = torch.log(torch.clamp(probs, 1e-8, 1.0))
                q1_pi = q1(s_b)
                q2_pi = q2(s_b)
                q_pi = torch.min(q1_pi, q2_pi)
                alpha = log_alpha.exp() if learn_alpha else torch.tensor(init_alpha, device=device)
                loss_pi = (probs * (alpha * logp - q_pi)).sum(dim=-1).mean()
                opt_pi.zero_grad(); loss_pi.backward(); opt_pi.step()
                if debug:
                    dbg(f"t={t} upd={upd+1} Actor: loss_pi={loss_pi.item():.6f}")

                # Temperature loss (optional)
                if learn_alpha and opt_alpha is not None:
                    ent = discrete_entropy(probs).detach()
                    loss_alpha = -(log_alpha * (target_entropy - ent).detach()).mean()
                    opt_alpha.zero_grad(); loss_alpha.backward(); opt_alpha.step()
                    dbg(f"t={t} upd={upd+1} Alpha: loss_alpha={loss_alpha.item():.6f}, alpha={float(log_alpha.exp().item()):.6f}")

                # Soft update of target networks
                soft_update(tgt_q1, q1, tau)
                soft_update(tgt_q2, q2, tau)
            if debug:
                dbg(f"t={t} 已完成 {max(1, updates_per_step)} 次梯度更新")

        # Episode handling (DummyEnv never ends; reset occasionally)
        if t % 200 == 0:
            state = env.reset()
            ep_ret = 0.0
            dbg(f"t={t} 周期性重置环境")

    # Save artifacts into run_dir
    torch.save(policy.state_dict(), run_dir / "policy.pt")
    torch.save(q1.state_dict(), run_dir / "q1.pt")
    torch.save(q2.state_dict(), run_dir / "q2.pt")
    # Save effective config snapshot
    try:
        snap = dict(cfg)
        snap.update({"device_resolved": str(device), "resume_from": str(run_dir) if resume else None})
        (run_dir / "config.snapshot.json").write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    # Minimal console note (no logger to keep dependencies minimal)
    print(f"Training finished. Prefilled: {prefilled} samples. Artifacts at: {run_dir}")


if __name__ == "__main__":
    train()










