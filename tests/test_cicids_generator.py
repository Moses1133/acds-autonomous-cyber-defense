"""Quick sanity check for CICIDSAttackGenerator."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.attacks.cicids import CICIDSAttackGenerator

gen = CICIDSAttackGenerator()
print(gen.summary())
print()

rng = np.random.default_rng(42)
print("Sample damage values per group (10 each):")
for group in gen.profiles.keys():
    samples = [round(gen.sample_damage(group, rng), 3) for _ in range(10)]
    print(f"  {group:14s} {samples}")
print()

print("Sample group choices by stage (10 each):")
for stage in [1, 2, 3]:
    picks = [gen.sample_group(stage, rng) for _ in range(10)]
    from collections import Counter
    print(f"  stage {stage}: {dict(Counter(picks))}")

print("\nCICIDS generator works.")
