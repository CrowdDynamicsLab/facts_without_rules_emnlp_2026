Facts Without Rules benchmark/testbed data bundle

Contents
- data/bound_handoff_phase2.json: main 36-scenario benchmark/testbed with upstream transcripts, handoff surfaces, downstream tasks, boundary markers, and operational facts.
- data/phase2_scenario_gold_metadata.json: per-scenario gold metadata used by evaluators and analysis scripts.
- data/bound_handoff_seed.json: 6-scenario seed version retained for inspection and lightweight smoke tests.

Excluded on purpose
- No model outputs.
- No judge outputs.
- No downstream evaluation outputs.
- No experiment result tables or summaries.
- No API keys or machine-specific paths.

Use this folder with cleanup_repro_code/ by copying or symlinking these files into dataset/data/ in a working checkout.
