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
import re
import time as _pytime
import urllib.request
import urllib.error
import json as _json
# 确保仓库根与 my_scripts 可被导入
try:
    _P = Path(__file__).resolve()
    # 默认指向 three-level-up 作为仓库根（.../repo_root）
    _REPO_ROOT = _P.parents[3]
    # 若目录层级与预期不符，回退到两级（.../lbopb）再探测上一层
    if not (_REPO_ROOT / 'my_scripts').is_dir():
        _REPO_ROOT = _P.parents[2]
        if not (_REPO_ROOT / 'my_scripts').is_dir() and len(_P.parents) > 3:
            _REPO_ROOT = _P.parents[3]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    _MS = _REPO_ROOT / "my_scripts"
    if _MS.is_dir() and str(_MS) not in sys.path:
        sys.path.insert(0, str(_MS))
except Exception:
    pass
try:
    # 优先使用集中封装的 my_scripts.gemini_client
    from my_scripts.gemini_client import generate_gemini_content as _gemini_generate_central
except Exception as _e:
    import sys as _sys
    _sys.stderr.write(f"[WARN] import my_scripts.gemini_client failed: {_e}\n")
    _gemini_generate_central = None

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
    # 每步打印：默认开启；可通过 env/config 调整
    log_every_step = bool(os.environ.get("RLSAC_LOG_EVERY_STEP", "1") or cfg.get("log_every_step", True))
    log_to_file = bool(cfg.get("log_to_file", True))

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

    class RunLogger:
        def __init__(self, path: Path, append: bool = False):
            self.path = path
            self.f = open(path, 'a' if append else 'w', encoding='utf-8', newline='')

        def write_line(self, text: str) -> None:
            try:
                # Ensure CRLF endings explicitly
                self.f.write(text + "\r\n")
                self.f.flush()
            except Exception:
                pass

        def close(self) -> None:
            try:
                self.f.flush()
                self.f.close()
            except Exception:
                pass

    def _gen_report_from_log(log_file: Path, out_md: Path) -> None:
        try:
            text = log_file.read_text(encoding='utf-8')
        except Exception:
            return
        pat = re.compile(r"^\[STEP\s+(?P<t>\d+)\]\s+a=(?P<a>-?\d+)\s+r=(?P<r>-?\d+\.?\d*)\s+d=(?P<d>True|False)\s+buf=(?P<buf>\d+)\s+upd=(?P<upd>\d+)\s+loss_q1=(?P<lq1>[-\.\d]+)\s+loss_q2=(?P<lq2>[-\.\d]+)\s+loss_pi=(?P<lpi>[-\.\d]+)\s+alpha=(?P<alpha>[-\.\d]+)$",
                             re.MULTILINE)
        rewards = []
        updates = 0
        last = {}
        for m in pat.finditer(text):
            r = float(m.group('r'))
            rewards.append(r)
            updates += int(m.group('upd'))
            last = {
                't': int(m.group('t')),
                'a': int(m.group('a')),
                'r': r,
                'd': m.group('d'),
                'buf': int(m.group('buf')),
                'upd': int(m.group('upd')),
                'lq1': m.group('lq1'),
                'lq2': m.group('lq2'),
                'lpi': m.group('lpi'),
                'alpha': m.group('alpha'),
            }
        mean_r = (sum(rewards) / len(rewards)) if rewards else 0.0
        last100 = (sum(rewards[-100:]) / len(rewards[-100:])) if len(rewards) >= 1 else 0.0
        lines = []
        lines.append(f"# 训练报告 (run: {log_file.parent.name})")
        lines.append("")
        lines.append(f"- 日期：{_pytime.strftime('%Y-%m-%d %H:%M:%S', _pytime.localtime())}")
        lines.append(f"- 设备：{device}")
        lines.append("")
        lines.append("## 概览")
        lines.append(f"- 总步数：{len(rewards)}")
        lines.append(f"- 总更新轮数：{updates}")
        lines.append(f"- 平均奖励：{mean_r:.6f}")
        lines.append(f"- 近100步平均奖励：{last100:.6f}")
        if last:
            lines.append(f"- 最后一次记录：t={last['t']} buf={last['buf']} upd={last['upd']} alpha={last['alpha']}")
            lines.append(f"  - loss_q1={last['lq1']} loss_q2={last['lq2']} loss_pi={last['lpi']}")
        lines.append("")
        lines.append("## 超参数")
        for k in [
            'learning_rate_actor','learning_rate_critic','gamma','tau','buffer_size','batch_size',
            'alpha','learn_alpha','target_entropy','total_steps','start_steps','update_after','update_every','updates_per_step','minibatch_floor'
        ]:
            if k in cfg:
                lines.append(f"- {k}: {cfg.get(k)}")
        # Ensure CRLF
        content = ("\n".join(lines)).replace("\r\n","\n").replace("\n","\r\n")
        out_md.write_text(content, encoding='utf-8')

    def _gemini_generate(prompt: str, *, api_key: str, model: str | None = None) -> str:
        # 如已安装集中封装，直接调用（模型优先取 GEMINI_MODEL）
        if _gemini_generate_central is not None:
            return _gemini_generate_central(prompt, api_key=api_key, model=model)
        # 本地兜底：从环境变量解析模型
        if not model:
            model = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro-latest")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "x-goog-api-key": api_key,
        }
        body = {"contents": [{"parts": [{"text": prompt}]}]}
        data = _json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                jo = _json.loads(raw)
        except urllib.error.HTTPError as e:
            try:
                err = e.read().decode("utf-8", errors="replace")
            except Exception:
                err = str(e)
            return f"[Gemini HTTPError] {e.code}: {err}"
        except Exception as e:
            return f"[Gemini Error] {e}"
        try:
            cands = jo.get("candidates", [])
            if not cands:
                return _json.dumps(jo, ensure_ascii=False)
            parts = cands[0].get("content", {}).get("parts", [])
            texts = [p.get("text") for p in parts if p.get("text")]
            return "\n".join(texts) if texts else _json.dumps(jo, ensure_ascii=False)
        except Exception:
            return _json.dumps(jo, ensure_ascii=False)

    def _gen_train_html_report(log_file: Path, cfg: Dict[str, Any], out_html: Path) -> None:
        # 解析日志，构造时间序列
        try:
            text = log_file.read_text(encoding='utf-8')
        except Exception:
            text = ''
        pat = re.compile(r"^\[STEP\s+(?P<t>\d+)\]\s+a=(?P<a>-?\d+)\s+r=(?P<r>-?\d+\.?\d*)\s+d=(?P<d>True|False)\s+buf=(?P<buf>\d+)\s+upd=(?P<upd>\d+)\s+loss_q1=(?P<lq1>[-\.\d]+)\s+loss_q2=(?P<lq2>[-\.\d]+)\s+loss_pi=(?P<lpi>[-\.\d]+)\s+alpha=(?P<alpha>[-\.\d]+)$",
                             re.MULTILINE)
        steps: list[int] = []
        rewards: list[float] = []
        lq1_s: list[float] = []
        lq2_s: list[float] = []
        lpi_s: list[float] = []
        alpha_s: list[float] = []
        for m in pat.finditer(text):
            try:
                steps.append(int(m.group('t')))
                rewards.append(float(m.group('r')))
                if m.group('lq1') not in ('-', ''):
                    lq1_s.append(float(m.group('lq1')))
                if m.group('lq2') not in ('-', ''):
                    lq2_s.append(float(m.group('lq2')))
                if m.group('lpi') not in ('-', ''):
                    lpi_s.append(float(m.group('lpi')))
                if m.group('alpha') not in ('-', ''):
                    alpha_s.append(float(m.group('alpha')))
            except Exception:
                continue
        mean_r = (sum(rewards) / len(rewards)) if rewards else 0.0
        last100 = (sum(rewards[-100:]) / max(1, len(rewards[-100:]))) if rewards else 0.0
        last_t = steps[-1] if steps else 0
        # Gemini 自动评价（若配置了 API Key）
        gemini_api = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        gemini_text: str | None = None
        if gemini_api and not os.environ.get('NO_GEMINI'):
            overview = {
                'total_steps': len(rewards),
                'updates_sum': None,
                'avg_reward': mean_r,
                'avg_reward_100': last100,
                'last_step': last_t,
                'cfg_brief': {k: cfg.get(k) for k in ['learning_rate_actor','learning_rate_critic','gamma','tau','batch_size','buffer_size','alpha','learn_alpha','target_entropy','updates_per_step'] if k in cfg}
            }
            prompt = (
                "请作为强化学习/DeepRL 专家，对下述 SAC 训练运行进行结构化诊断并给出改进建议。"
                "请返回 JSON，包含 fields: summary, stability_score, efficiency_score, risk_flags, signals, top_actions, caveats, confidence。\n"
                "训练概览(JSON)：\n" + _json.dumps(overview, ensure_ascii=False)
            )
            try:
                gemini_text = _gemini_generate(prompt, api_key=gemini_api)
            except Exception:
                gemini_text = None

        # 生成 HTML（深色主题 + 图表）
        L: list[str] = []; A = L.append
        A("<!DOCTYPE html>")
        A("<html lang=\"zh-CN\">")
        A("<head>")
        A("  <meta charset=\"utf-8\" />")
        A("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />")
        A("  <title>SAC 训练报告 · {}".format(log_file.parent.name))
        A("  <script src=\"https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js\"></script>")
        A("  <style>")
        A("    :root{--bg:#0b1020;--bg2:#0f1830;--card:#141a2a;--txt:#e8efff;--muted:#9fb0d0;--good:#27ae60;--warn:#f39c12;--bad:#e74c3c}")
        A("    body{margin:0;font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:linear-gradient(180deg,var(--bg),var(--bg2));color:var(--txt)}")
        A("    .container{max-width:1120px;margin:0 auto;padding:24px}")
        A("    .hero{padding:56px 24px 24px}")
        A("    .card{background:#141a2a;border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:16px;margin:12px 0}")
        A("    .muted{color:#9fb0d0}")
        A("    pre{white-space:pre-wrap;background:#0d1322;border:1px solid rgba(255,255,255,.08);padding:12px;border-radius:8px}")
        A("    /* 兼容性：若背景未渲染（某些预览器），用浅色主题避免看起来空白 */")
        A("    .no-bg body{background:#ffffff !important;color:#111 !important}")
        A("    .no-bg .card{background:#ffffff !important;border:1px solid #dddddd}")
        A("    .no-bg .muted{color:#555555 !important}")
        A("    @media print{body{background:#ffffff !important;color:#111 !important}.card{background:#fff !important;border:1px solid #ddd}.muted{color:#555 !important}}");
        A("  </style>")
        # 当脚本被禁用时，使用 noscript 强制浅色主题，避免白底白字看起来像空白
        A("  <noscript><style>body{background:#fff !important;color:#111 !important}.card{background:#fff !important;border:1px solid #ddd}.muted{color:#555 !important}</style></noscript>")
        A("</head>")
        A("<body>")
        A("  <header class=\"hero\"><div class=\"container\">")
        A("    <h1>SAC 训练报告 · {}</h1>".format(log_file.parent.name))
        A("    <div class=\"muted\">本页由训练脚本自动生成，并尝试调用 Gemini 进行自动评价（如已配置密钥）。</div>")
        A("  </div></header>")
        A("  <main class=\"container\">")
        A("    <script>")
        A("    (function(){try{var bg=getComputedStyle(document.body).backgroundImage;if(!bg||bg==='none'){document.documentElement.classList.add('no-bg');}}catch(e){}})();")
        A("    </script>")
        # 概览
        A("    <section class=\"card\">")
        A("      <h2>概览</h2>")
        A("      <ul>")
        A("        <li>总步数：{}</li>".format(len(rewards)))
        A("        <li>平均奖励：{:.6f}</li>".format(mean_r))
        A("        <li>近100步平均奖励：{:.6f}</li>".format(last100))
        A("        <li>最后步：{}</li>".format(last_t))
        A("      </ul>")
        A("    </section>")
        # 图表
        A("    <section class=\"card\">")
        A("      <h2>奖励趋势</h2>")
        A("      <canvas id=\"reward_chart\" height=\"160\"></canvas>")
        A("    </section>")
        # LLM 评价
        A("    <section class=\"card\">")
        A("      <h2>Gemini 评价结果</h2>")
        if gemini_text:
            esc = gemini_text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            A("      <pre>{}</pre>".format(esc))
        elif not gemini_api:
            A("      <p class=\"muted\">未检测到 GEMINI_API_KEY/GOOGLE_API_KEY，已跳过自动评价。</p>")
        else:
            A("      <p class=\"muted\">调用 Gemini 失败或无响应。</p>")
        A("    </section>")
        # 超参数
        A("    <section class=\"card\">")
        A("      <h2>超参数</h2>")
        A("      <pre>{}</pre>".format(_json.dumps(cfg, ensure_ascii=False, indent=2)))
        A("    </section>")
        # 图表脚本
        A("    <script>")
        A("const STEPS = {}".format(steps))
        A("const REWARDS = {}".format([round(x, 6) for x in rewards]))
        A("const ctx = document.getElementById('reward_chart').getContext('2d');")
        A("new Chart(ctx, {type:'line', data:{labels:STEPS, datasets:[{label:'reward', data:REWARDS, borderColor:'#36c', pointRadius:0, tension:0.15}]}, options:{plugins:{legend:{labels:{color:'#cfe0ff'}}}, scales:{x:{ticks:{color:'#9fb0d0', display:false}}, y:{ticks:{color:'#9fb0d0'}, grid:{color:'rgba(255,255,255,.08)'}}}}});")
        A("    </script>")
        A("  </main>")
        A("</body>")
        A("</html>")
        content = ("\n".join(L)).replace("\r\n","\n").replace("\n","\r\n") + "\r\n"
        out_html.write_text(content, encoding='utf-8')

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
    dbg(
        "超参数: lr_actor={:.1e}, lr_critic={:.1e}, gamma={}, tau={}, buffer_size={}, batch_size={}, "
        "target_entropy={}, learn_alpha={}, alpha_init={}, total_steps={}, start_steps={}, update_after={}, update_every={}, updates_per_step={}, minibatch_floor={}".format(
            lr_actor, lr_critic, gamma, tau, buffer_size, batch_size, target_entropy, learn_alpha, init_alpha,
            total_steps, start_steps, update_after, update_every, updates_per_step, minibatch_floor
        )
    )

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

    # Initialize logger
    logger: RunLogger | None = None
    log_path = run_dir / "train.log"
    if log_to_file:
        logger = RunLogger(log_path, append=resume)
        logger.write_line(f"# TRAIN START { _pytime.strftime('%Y-%m-%d %H:%M:%S', _pytime.localtime()) }")
        logger.write_line(f"device={device}")
        logger.write_line(f"config={json.dumps(cfg, ensure_ascii=False)}")

    # Networks
    policy = DiscretePolicy(obs_dim, n_actions).to(device)
    q1 = QNetwork(obs_dim, n_actions).to(device)
    q2 = QNetwork(obs_dim, n_actions).to(device)
    tgt_q1 = QNetwork(obs_dim, n_actions).to(device)
    tgt_q2 = QNetwork(obs_dim, n_actions).to(device)
    dbg("网络已创建: policy/q1/q2 及目标网络")
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
    dbg("优化器已创建: opt_pi/opt_q1/opt_q2")

    # Temperature parameter alpha (either fixed or learnable)
    log_alpha = torch.tensor(float(torch.log(torch.tensor(init_alpha))), requires_grad=learn_alpha, device=device)
    opt_alpha = torch.optim.Adam([log_alpha], lr=lr_actor) if learn_alpha else None
    dbg(f"温度系数: learn_alpha={learn_alpha}, alpha_init={init_alpha}")

    # Replay buffer
    buf = ReplayBuffer(buffer_size, obs_dim)
    data_path = Path(data_json) if data_json else (mod_dir / "operator_crosswalk_train.json")
    prefilled = _prefill_from_json(buf, data_path, obs_dim)
    dbg(f"预填充回放池: 来自 {data_path}, 条目数={prefilled}")

    state = env.reset()
    ep_ret = 0.0
    last_q1 = None
    last_q2 = None
    last_pi = None
    last_alpha = float(log_alpha.exp().item()) if learn_alpha else init_alpha
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
        updates_done = 0
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
                last_q1 = float(loss_q1.item())
                last_q2 = float(loss_q2.item())

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
                last_pi = float(loss_pi.item())

                # Temperature loss (optional)
                if learn_alpha and opt_alpha is not None:
                    ent = discrete_entropy(probs).detach()
                    loss_alpha = -(log_alpha * (target_entropy - ent).detach()).mean()
                    opt_alpha.zero_grad(); loss_alpha.backward(); opt_alpha.step()
                    dbg(f"t={t} upd={upd+1} Alpha: loss_alpha={loss_alpha.item():.6f}, alpha={float(log_alpha.exp().item()):.6f}")
                    last_alpha = float(log_alpha.exp().item())

                # Soft update of target networks
                soft_update(tgt_q1, q1, tau)
                soft_update(tgt_q2, q2, tau)
            if debug:
                dbg(f"t={t} 已完成 {max(1, updates_per_step)} 次梯度更新")
            updates_done = max(1, updates_per_step)

        # 每步打印汇总
        step_log(
            "[STEP {t}] a={a} r={r:.6f} d={d} buf={buf} upd={upd} "
            "loss_q1={lq1} loss_q2={lq2} loss_pi={lpi} alpha={alpha}".format(
                t=t, a=action, r=r, d=d, buf=len(buf), upd=updates_done,
                lq1=("{:.6f}".format(last_q1) if last_q1 is not None else "-"),
                lq2=("{:.6f}".format(last_q2) if last_q2 is not None else "-"),
                lpi=("{:.6f}".format(last_pi) if last_pi is not None else "-"),
                alpha=("{:.6f}".format(last_alpha) if last_alpha is not None else "-")
            )
        )

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
    if logger:
        logger.write_line("# TRAIN END")
        logger.close()
        # Generate markdown report
        try:
            _gen_report_from_log(log_path, run_dir / "train_report.md")
        except Exception:
            pass
        # 生成美化 HTML 报告，并尝试自动请求 Gemini
        try:
            _gen_train_html_report(log_path, cfg, run_dir / "train_report.html")
        except Exception as e:
            print(f"Warning: failed to generate HTML report: {e}")

        # 输出“模型使用脚本 + 使用说明”到当前训练目录
        try:
            infer_py = run_dir / "run_infer.py"
            infer_md = run_dir / "MODEL_USAGE.md"
            _infer_src = (
                "# SPDX-License-Identifier: GPL-3.0-only\r\n"
                "# Copyright (C) 2010- The GROMACS Authors\r\n"
                "# Copyright (C) 2025 GaoZheng\r\n\r\n"
                "\"\"\"\r\n"
                "轻量推理脚本：加载本目录的 policy.pt 并在 DummyEnv 上滚动演示。\r\n"
                "用法：\r\n"
                "  python run_infer.py --steps 200 --episodes 1 --device cpu\r\n"
                "可选：--model policy.pt --csv infer_trace.csv\r\n"
                "\"\"\"\r\n\r\n"
                "from __future__ import annotations\r\n"
                "import argparse, os, sys, csv\r\n"
                "from pathlib import Path\r\n"
                "import torch\r\n"
                "# 尝试确保可以导入包\r\n"
                "_P = Path(__file__).resolve()\r\n"
                "for up in [2,3,4]:\r\n"
                "    root = _P.parents[up] if len(_P.parents) > up else _P.parent\r\n"
                "    if (root / 'lbopb').is_dir():\r\n"
                "        if str(root) not in sys.path: sys.path.insert(0, str(root))\r\n"
                "        break\r\n"
                "from lbopb.src.rlsac.env import DummyEnv\r\n"
                "from lbopb.src.rlsac.models import DiscretePolicy\r\n\r\n"
                "def main():\r\n"
                "    ap = argparse.ArgumentParser(description='SAC policy inference (DummyEnv)')\r\n"
                "    ap.add_argument('--model', default='policy.pt')\r\n"
                "    ap.add_argument('--steps', type=int, default=200)\r\n"
                "    ap.add_argument('--episodes', type=int, default=1)\r\n"
                "    ap.add_argument('--device', default='cpu')\r\n"
                "    ap.add_argument('--csv', default='')\r\n"
                "    args = ap.parse_args()\r\n\r\n"
                "    device = torch.device(args.device)\r\n"
                "    env = DummyEnv()\r\n"
                "    obs_dim = env.observation_space.shape[0]\r\n"
                "    n_actions = env.action_space.high - env.action_space.low\r\n"
                "    policy = DiscretePolicy(obs_dim, n_actions).to(device)\r\n"
                "    ckpt = torch.load(args.model, map_location=device)\r\n"
                "    policy.load_state_dict(ckpt)\r\n"
                "    policy.eval()\r\n"
                "    writer = None\r\n"
                "    if args.csv:\r\n"
                "        f = open(args.csv, 'w', encoding='utf-8', newline='')\r\n"
                "        writer = csv.writer(f)\r\n"
                "        writer.writerow(['t','action','reward','done'])\r\n"
                "    for ep in range(args.episodes):\r\n"
                "        s = env.reset()\r\n"
                "        ep_ret = 0.0\r\n"
                "        for t in range(1, args.steps + 1):\r\n"
                "            with torch.no_grad():\r\n"
                "                logits, probs = policy(torch.tensor([s], dtype=torch.float32).to(device))\r\n"
                "                a = int(torch.argmax(probs, dim=-1).item())\r\n"
                "            s2, r, d, _ = env.step(a)\r\n"
                "            ep_ret += float(r)\r\n"
                "            print(f'[INFER] ep={{ep+1}} t={{t}} a={{a}} r={{r:.6f}} done={{d}}')\r\n"
                "            if writer: writer.writerow([t, a, float(r), bool(d)])\r\n"
                "            s = s2\r\n"
                "            if d: break\r\n"
                "        print(f'[INFER] episode_return={{ep_ret:.6f}}')\r\n"
                "    if writer: writer.writerow([]); writer = None\r\n"
                "\r\n"
                "if __name__ == '__main__':\r\n"
                "    main()\r\n"
            )
            infer_py.write_text(_infer_src, encoding='utf-8', newline='\r\n')

            _infer_md = (
                f"# 模型使用说明 (run: {run_dir.name})\r\n\r\n"
                "本目录包含训练得到的策略网络（policy.pt）与推理脚本 run_infer.py，可用于在 DummyEnv 上完成快速滚动演示。\r\n\r\n"
                "## 准备环境\r\n\r\n"
                "```powershell\r\n"
                "# 可选：创建虚拟环境\r\n"
                "python -m venv .venv\r\n"
                "# Windows PowerShell 激活\r\n"
                ".\\.venv\\Scripts\\Activate.ps1\r\n"
                "# 安装依赖\r\n"
                "pip install -r requirement.txt\r\n"
                "```\r\n\r\n"
                "## 运行推理\r\n\r\n"
                "```powershell\r\n"
                "python run_infer.py --steps 200 --episodes 1 --device cpu\r\n"
                "# 或指定权重/输出 CSV\r\n"
                "python run_infer.py --model policy.pt --steps 500 --csv infer_trace.csv\r\n"
                "```\r\n\r\n"
                "## 说明\r\n\r\n"
                "- 该脚本使用 lbopb/src/rlsac 下的 DummyEnv 进行演示；\r\n"
                "- 若需在自定义环境上推理，请替换 env 与状态/动作维度；\r\n"
                "- 本推理脚本仅用于连通性测试与演示，不代表任何生产评估。\r\n"
            )
            infer_md.write_text(_infer_md, encoding='utf-8', newline='\r\n')
        except Exception as e:
            print(f"Warning: failed to write inference package: {e}")


if __name__ == "__main__":
    train()











