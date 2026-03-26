# Objective
Improve the target module using small controlled experiments.

# Editable files
You may modify only:
- research_sandbox/candidate_module.py

# Read-only files
- research_sandbox/eval.py
- research_sandbox/results.tsv
- all other files in the repo

# Primary metric
Maximize: score

# Guardrails
- no exceptions
- deterministic output
- runtime under 60 seconds
- do not modify files outside research_sandbox

# Workflow
1. Read candidate_module.py and eval.py
2. Explain the current logic before changing anything
3. Make only one small change at a time
4. Run eval.py after each change
5. Keep the change only if score improves
6. Append a short line to results.tsv with result summary
7. If score does not improve, revert the change
