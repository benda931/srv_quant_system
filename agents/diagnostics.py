"""
agents/diagnostics.py
=======================
Agent health diagnostics — test all agents and report status.

Usage:
    python agents/diagnostics.py              # Import test only (fast)
    python agents/diagnostics.py --run        # Actually run each agent (slow)
    python agents/diagnostics.py --agent auto_improve  # Test one agent
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")


def main():
    from agents.shared.agent_interface import AGENT_REGISTRY, diagnose_agents, run_agent

    if "--agent" in sys.argv:
        idx = sys.argv.index("--agent")
        if idx + 1 < len(sys.argv):
            name = sys.argv[idx + 1]
            print(f"Running agent: {name}")
            result = run_agent(name)
            print(f"  Status:   {result.status}")
            print(f"  Duration: {result.duration_s:.1f}s")
            print(f"  Import:   {'OK' if result.import_ok else 'FAIL'}")
            print(f"  Run:      {'OK' if result.run_ok else 'FAIL'}")
            if result.error:
                print(f"  Error:    {result.error}")
            if result.metrics:
                for k, v in result.metrics.items():
                    if k not in ("status",):
                        print(f"  {k}: {v}")
            return

    do_run = "--run" in sys.argv

    print(f"{'='*60}")
    print(f"  Agent Diagnostics — {len(AGENT_REGISTRY)} registered agents")
    print(f"  Mode: {'IMPORT + RUN' if do_run else 'IMPORT ONLY'}")
    print(f"{'='*60}")
    print()

    results = diagnose_agents(run_test=do_run)

    n_import_ok = 0
    n_import_fail = 0
    n_run_ok = 0
    n_run_fail = 0

    for name, diag in sorted(results.items()):
        manifest = AGENT_REGISTRY[name]

        if diag["import_ok"]:
            n_import_ok += 1
            import_icon = "✓"
        else:
            n_import_fail += 1
            import_icon = "✗"

        if do_run:
            if diag.get("run_ok"):
                n_run_ok += 1
                run_icon = "✓"
            elif diag.get("run_ok") is False:
                n_run_fail += 1
                run_icon = "✗"
            else:
                run_icon = "?"
            run_info = f"run={run_icon} {diag.get('duration', 0):.0f}s"
        else:
            run_info = ""

        err = f" | {diag['error'][:50]}" if diag.get("error") else ""
        print(f"  {import_icon} {name:<25} import={import_icon} {run_info}{err}")
        print(f"    Mandate: {manifest.mandate[:60]}")
        print(f"    Writes:  {', '.join(manifest.writes_to[:2])}")
        print()

    print(f"{'='*60}")
    print(f"  Import: {n_import_ok}/{n_import_ok + n_import_fail} OK")
    if do_run:
        print(f"  Run:    {n_run_ok}/{n_run_ok + n_run_fail} OK")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
