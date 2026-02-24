# Emergence

Bootstrapping a geometric world model from a pretrained LLM.

## Intent

Current LLMs are statistical interpolators over token sequences. They can describe physics without their internal representations being physically consistent. The geometry of their embedding space is an emergent accident of training, not a designed property. This project tests whether a designed geometric idea space, continuously updated via relational annotation, can produce representations that are causally and semantically consistent by construction rather than by coincidence.

The hypothesis: if every concept has a position in a shared coordinate space, and every relation between concepts is typed and persistent, the accumulated web will develop internal consistency that generalizes. New concepts placed relative to existing anchors will inherit correct relational properties without being explicitly taught.

This is distinct from knowledge graphs (no geometry), word embeddings (static, untyped, non-causal), and neurosymbolic approaches (brittle rule engines). The geometry is the representation.

## Architecture

A pretrained LLM (`qwen2.5-coder:14b-instruct` via Ollama) acts as a relational annotator, not the world model itself. It receives a factual sentence plus the current world state and outputs:

- XYZ positions for new or updated entities (0–1 scale, relative to existing anchors)
- Typed relations between entities (e.g. `cause`, `effect`, `part_of`, `located_in`)
- A next sentence to evaluate (local continuation)
- A jump sentence from an unrelated domain (forced exploration)

The annotator output is merged into a persistent `world_state.json`. The annotator is stateless; all geometry lives in the state file and is passed back in on every call. The LLM is a tool, not the substrate.

## Why This Loop

Self-generating sentences address the corpus problem: instead of curating training data, the model explores concept space autonomously. Alternating `next` and `jump` implements exploitation and exploration. Local coherence limits geometric drift; forced domain jumps prevent the web from collapsing into a single semantic cluster.

Positions self-anchor: once enough entities exist, new placements are triangulated against multiple reference points rather than placed arbitrarily. Early iterations are noisy; the geometry stabilizes as anchor density increases.

## Observed Behavior

Physical and concrete entities cluster at low Z. Abstract and causal entities (forces, processes) cluster at high Z. This axis emerges without explicit instruction; the model infers it from the semantic content of sentences. Geographic containment relations (e.g. Paris in France) yield close XY positions. Cross-domain concepts (physical objects vs. geographic entities) share Z-level when they share the property of being concrete, regardless of domain.

Whether these clusters reflect genuine semantic structure or surface correlations in the annotator’s training data is the open question this project is designed to answer.

## State Compression

Context budget is finite. Above 2000 characters the state drops relation history and passes entity positions only. Positions are the load-bearing structure; relations can be partially reconstructed from geometry. This is a known lossy tradeoff. A future version should maintain a compressed relation index separately from the full state.

## Files

| File or directory   | Purpose                          |
|---------------------|----------------------------------|
| `main.py`           | Recursive annotation loop       |
| `world_state.json`  | Persistent geometric world model |
| `log/`              | Per-run timestamped logs        |

## Usage

```bash
pip install requests
ollama pull qwen2.5-coder:14b-instruct
python main.py
tail -f log/<timestamp>.log
```

To visualize accumulated state, use the provided script:

```bash
pip install matplotlib
python visualize.py
```

Or load and plot manually:

```python
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

s = json.load(open("world_state.json"))
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for name, pos in s["entities"].items():
    ax.scatter(pos["x"], pos["y"], pos["z"])
    ax.text(pos["x"], pos["y"], pos["z"], name, size=7)
plt.show()
```

## Open Questions

- Do relation types cluster geometrically (e.g. do all `cause` relations point in a consistent direction)?
- Does the geometry generalize? Can unseen concepts be placed correctly by analogy?
- At what entity count does the web become self-consistent enough to reject bad placements?
- Can the annotator role be replaced by a smaller model trained on the accumulated state?

The last question is the long-term goal.
