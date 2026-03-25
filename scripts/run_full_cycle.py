"""Full Agent Cycle — run all 10 agents in dependency order."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

agents = [
    ("1. Data Scout",            [sys.executable, "-m", "agents.data_scout", "--once"]),
    ("2. Regime Forecaster",     [sys.executable, "-m", "agents.regime_forecaster", "--once"]),
    ("3. Risk Guardian",         [sys.executable, "-m", "agents.risk_guardian", "--once"]),
    ("4. Alpha Decay",           [sys.executable, "-m", "agents.alpha_decay", "--once"]),
    ("5. Portfolio Construction", [sys.executable, "-m", "agents.portfolio_construction", "--once"]),
    ("6. Auto-Improve (dry)",    [sys.executable, "-m", "agents.auto_improve", "--cycle", "--dry-run"]),
]


def main():
    print("=" * 70)
    print("FULL AGENT CYCLE — 10-Agent Hedge Fund System")
    print("=" * 70)

    results = {}
    for name, cmd in agents:
        print(f"\n{'─' * 50}")
        print(f"▶ {name}")
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=str(ROOT))
            elapsed = time.time() - t0
            ok = r.returncode == 0
            lines = [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]
            summary = lines[-1][:120] if lines else "no output"
            results[name] = {"ok": ok, "time": round(elapsed, 1)}
            status = "✅" if ok else "❌"
            print(f"  {status} {elapsed:.1f}s — {summary}")
            if not ok and r.stderr:
                err = r.stderr.strip().split("\n")[-1][:120]
                print(f"  Error: {err}")
        except subprocess.TimeoutExpired:
            results[name] = {"ok": False, "time": 180}
            print("  ⏰ TIMEOUT (180s)")
        except Exception as e:
            results[name] = {"ok": False, "time": 0}
            print(f"  ❌ {e}")

    print(f"\n{'=' * 70}")
    print("CYCLE SUMMARY")
    print(f"{'=' * 70}")
    ok_count = sum(1 for r in results.values() if r["ok"])
    total_time = sum(r["time"] for r in results.values())
    print(f"  Agents: {ok_count}/{len(results)} successful")
    print(f"  Total time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    for name, r in results.items():
        s = "✅" if r["ok"] else "❌"
        print(f"  {s} {name}: {r['time']:.0f}s")


if __name__ == "__main__":
    main()
