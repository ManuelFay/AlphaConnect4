# GNUBG Python Package Analysis

## Overview

**[gnubg-nn-pypi](https://github.com/reayd-falmouth/gnubg-nn-pypi)** is a Python 3 wrapper around GNU Backgammon's neural network engine. It provides position evaluation, move analysis, and board utilities.

- **PyPI**: `pip install gnubg`
- **Docs**: https://gnubg.readthedocs.io/en/latest/
- **License**: GPL-3.0 (same as GnuBG)
- **Python**: 3.8-3.13
- **Platforms**: Linux, macOS (Intel & ARM64), Windows

## How It Works

The package bundles pre-trained neural networks. When you call `gnubg.initnet()`, it loads the weights into memory. Then evaluation functions (`probabilities()`, `best_move()`) run positions through the neural net to produce outputs.

**Neural Network Architecture**:
- 250 input nodes (192 raw board + ~58 expert features)
- 128 hidden nodes
- 5 outputs (win%, gammon win%, BG win%, gammon loss%, BG loss%)
- 6 specialized networks: contact, crashed, race, bearoff, backcontain, over

---

## Key Capabilities for Our Project

### 1. Position ID Encoding/Decoding (CRITICAL)

```python
import gnubg

gnubg.initnet()  # Initialize engine (required once)

# Decode: Position ID -> Board
board = gnubg.board_from_position_id("4HPwATDgc/ABMA")
# Returns 2x25 array: [[X checkers], [O checkers]]

# Encode: Board -> Position ID
pos_id = gnubg.position_id(board)
# Returns "4HPwATDgc/ABMA"
```

**This solves our encoding problem completely.** We can use GnuBG position IDs as the interchange format.

### 2. Position Classification (BUILT-IN!)

```python
cls = gnubg.classify(board)

# Classification constants:
# gnubg.c_contact   - Contact position
# gnubg.c_race      - Pure race
# gnubg.c_crashed   - Crashed/crunched position
# gnubg.c_bearoff   - Bearoff position
# gnubg.c_over      - Game over
# gnubg.c_backcontain - Back containment

if cls == gnubg.c_race:
    print("Race position detected")
```

**Note**: This is a simpler classification than our 17-category taxonomy. We'd use this as a first-pass filter, then apply our finer-grained rules.

### 3. Win/Gammon Probabilities

```python
# 5-tuple: (win, win_gammon, win_bg, lose_gammon, lose_bg)
probs = gnubg.probabilities(board, gnubg.p_prune)
# (0.637, 0.182, 0.042, 0.103, 0.036)

# With rollout for more accuracy
probs = gnubg.rollout(board, ngames=1296)
```

**Use case**: Display equity/probabilities alongside our position classification.

### 4. Best Move Analysis

```python
# Basic: returns move as [(from, to), ...]
move = gnubg.best_move(board, die1=3, die2=1)
# [(8, 5), (6, 5)]

# Extended: with alternate moves and evaluations
move, new_board, resign, alternatives = gnubg.best_move(
    board, 3, 1,
    n=2,        # 2-ply search
    b=True,     # return resulting board
    r=True,     # return resignation advice
    list=True   # return alternative moves
)
```

**Use case**: Training mode - show best move after user attempts.

### 5. Legal Move Generation

```python
# All legal moves from position
legal = gnubg.moves(board, die1=3, die2=1, verbose=True)
# Returns all possible moves with resulting positions
```

**Use case**: Move validation, UI highlighting of legal moves.

### 6. Match Equity Tables

```python
from gnubg import equities

# X is 3-away, O is 2-away
equity = equities.value(3, 2)  # 0.638 = 63.8% for X
```

**Use case**: Match play training, cube decisions.

### 7. Cube Decision Support

```python
# Cubeful rollout with cube decisions
result = gnubg.cubeful_rollout(board, ngames=576, side='X', ply=0)
# Returns 13-value tuple with equity and cube metrics
```

---

## Board Representation

GnuBG uses a 2x25 array:
- `board[0]` = X's checkers (player on roll)
- `board[1]` = O's checkers
- Index 0 = bar
- Index 1-24 = points (1 = X's ace point, 24 = O's ace point)
- Index 25 = borne off (sometimes)

```python
# Starting position
board = gnubg.board_from_position_id("4HPwATDgc/ABMA")
# board[0] = [0, 0, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
# board[1] = [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, 0]
```

---

## What's Exposed vs What's Internal

### Exposed Through Python API

| Function | Description |
|----------|-------------|
| `board_from_position_id(id)` | Decode GNUID → 2x25 board array |
| `position_id(board)` | Encode board → GNUID |
| `classify(board)` | Basic classification (6 types) |
| `probabilities(board, ply)` | Win/gammon/BG percentages |
| `best_move(board, d1, d2, n)` | Optimal move at n-ply |
| `moves(board, d1, d2)` | All legal moves |
| `rollout(board, ngames)` | Monte Carlo simulation |
| `cubeful_rollout(...)` | Cube-aware rollout |
| `equities.value(x, o)` | Match equity table |
| `pub_eval_score(board)` | Heuristic position score |
| `bearoff_probabilities(id)` | Bearoff analysis |

### NOT Exposed (Internal to Neural Net)

The C code (`eval.c`) computes ~58 expert features via `CalculateHalfInputs()` for neural net input, but these are **not accessible from Python**:

| Feature Category | Examples |
|-----------------|----------|
| **Anchor detection** | Back anchor location, forward anchor position |
| **Pip calculations** | Pip count, pip loss from hits |
| **Hit probability** | Prob of hitting 1 checker, 2 checkers |
| **Escape probability** | Back checker escape chances |
| **Containment** | Opponent containment, player containment metrics |
| **Mobility** | Movement flexibility calculations |
| **Timing** | Backbone strength, timing metrics |
| **Position spread** | Moment calculations |
| **Entry probability** | Chances of entering from bar |

**Source**: These are computed in [`eval.c`](https://github.com/mormegil-cz/gnubg/blob/master/eval.c) via `CalculateHalfInputs()` with 40+ enumerated input constants (`I_OFF1` through `I_BACKG1`).

### What This Means For Us

| Task | Who Does It |
|------|-------------|
| Position encoding/decoding | gnubg |
| Win/gammon/BG probabilities | gnubg |
| Best move analysis | gnubg |
| Basic classification (6 types) | gnubg |
| **Pip count** | **Us** (trivial) |
| **Prime detection** | **Us** |
| **Anchor detection** | **Us** |
| **Timing proxies** | **Us** |
| **17-category classification** | **Us** |

**Note**: If we need the exact same features gnubg uses, we can reference the C source code in `eval.c` and port the logic to Python. The code is GPL-licensed and available at [mormegil-cz/gnubg](https://github.com/mormegil-cz/gnubg).

---

## Summary: What We Get For Free

| Feature | GnuBG Provides | Our Work |
|---------|---------------|----------|
| Position ID encode/decode | Yes | None needed |
| Basic classification (6 types) | Yes | Use as first pass |
| Fine-grained classification (17 types) | No | We build this |
| Win/gammon probabilities | Yes | None needed |
| Best move analysis | Yes | None needed |
| Legal move generation | Yes | None needed |
| Match equity tables | Yes | None needed |
| Pip count | No | Simple formula |
| Position features (primes, anchors) | No | We build (can reference C code) |
| Board visualization | No | We build (Svelte) |

---

## Integration Strategy

### Option A: GnuBG as Core Engine (Recommended)

```
User Input (Position ID or Screenshot)
         ↓
    GnuBG decode → Board array
         ↓
    GnuBG classify → Basic type (race/contact/bearoff/etc)
         ↓
    Our classifier → Fine-grained type (17 categories)
         ↓
    GnuBG probabilities → Win/gammon chances
         ↓
    Display in UI
```

**Pros**:
- Battle-tested position handling
- Neural net evaluation for free
- Cube/match equity built-in

**Cons**:
- GPL license (must open-source our code if distributed)
- C extension dependency (compilation on some platforms)

### Option B: GnuBG for Encoding Only

Use GnuBG just for position ID encoding/decoding, implement our own evaluation.

**Pros**: Simpler dependency, more control

**Cons**: No neural net evaluation, reinventing wheel

---

## Installation & Setup

```bash
# Install via pip
pip install gnubg

# Or via uv (our package manager)
uv add gnubg
```

```python
import gnubg

# MUST call initnet() before any other function
gnubg.initnet()

# Now ready to use
board = gnubg.board_from_position_id("4HPwATDgc/ABMA")
```

---

## Example: Full Position Analysis

```python
import gnubg

gnubg.initnet()

def analyze_position(position_id: str) -> dict:
    """Full analysis of a backgammon position."""

    board = gnubg.board_from_position_id(position_id)

    # Basic classification
    cls = gnubg.classify(board)
    cls_names = {
        gnubg.c_contact: "contact",
        gnubg.c_race: "race",
        gnubg.c_crashed: "crashed",
        gnubg.c_bearoff: "bearoff",
        gnubg.c_over: "over",
        gnubg.c_backcontain: "back_containment",
    }

    # Probabilities
    win, wg, wbg, lg, lbg = gnubg.probabilities(board, gnubg.p_prune)

    # Pip counts (compute from board)
    def pip_count(checkers):
        return sum(i * checkers[i] for i in range(1, 25))

    x_pips = pip_count(board[0])
    o_pips = pip_count(board[1])

    return {
        "position_id": position_id,
        "gnubg_class": cls_names.get(cls, "unknown"),
        "probabilities": {
            "win": win,
            "win_gammon": wg,
            "win_backgammon": wbg,
            "lose_gammon": lg,
            "lose_backgammon": lbg,
        },
        "equity": win + wg + wbg - lg - lbg,  # Simplified
        "pip_count": {"x": x_pips, "o": o_pips},
        "pip_diff": x_pips - o_pips,
    }
```

---

## Open Questions

1. **License implications**: GPL means if we distribute, we must open-source. Fine for personal project, consider for commercial use.

2. **Compilation**: The package has pre-built wheels for common platforms. May need to handle edge cases.

3. **Classification mapping**: GnuBG's 6 categories vs our 17. Need to map:
   - `c_contact` → could be blitz, prime, anchor, middlegame, etc.
   - `c_race` → our "pure_race"
   - `c_crashed` → our "crunched"
   - `c_bearoff` → our "late_game_contact" or "pure_race" depending on contact
   - `c_backcontain` → our "containment"

4. **Performance**: C extension should be fast. Benchmark if needed.

---

## Resources

- [GitHub Repository](https://github.com/reayd-falmouth/gnubg-nn-pypi)
- [PyPI Package](https://pypi.org/project/gnubg/)
- [Documentation](https://gnubg.readthedocs.io/en/latest/)
- [GNU Backgammon Manual](https://www.gnu.org/software/gnubg/manual/)
