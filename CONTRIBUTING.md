# Contributing

Run `pip install -e '.[test]'` and `pytest -q` before proposing changes. Include the configuration, seed, manifest checksum, software versions and measurement scope with numerical claims. Use focused changes and tests for attention math, mask semantics and split integrity. Keep source scene groups intact across train/validation/test.

Use a new run directory for each experiment. State whether a result uses synthetic fixtures, a short real-data execution check or the full research protocol. Compare baselines with the same annotations, resolution, split and metric definitions. Document new architecture choices in the method map rather than implying they were specified by the manuscript.

Keep research data, credentials, manuscript drafts, checkpoints and personal/location metadata outside commits. Cite upstream implementations and dataset terms when introducing dependencies or adapters. The current model is image-based and binary; extensions to subtype classification or multimodal fusion need their own data contract and evaluation protocol.
