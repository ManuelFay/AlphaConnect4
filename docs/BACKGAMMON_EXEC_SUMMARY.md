# Backgammon MCTS Project - Executive Summary

## TL;DR

**Don't reinvent the wheel.** Use `gnubg` for the engine + rules, focus your effort on the MCTS algorithm and a modern UX.

---

## What Already Exists

### Engine & Rules: gnubg-nn-pypi
```bash
pip install gnubg
```

This gives you **for free**:
- Position encoding/decoding (GNUID format)
- Legal move generation: `gnubg.moves(board, die1, die2)`
- Best move at N-ply: `gnubg.best_move(board, d1, d2, n=2)`
- Win/gammon probabilities (neural net eval)
- World-class AI to benchmark against

**Bottom line**: No need to implement backgammon rules from scratch.

---

## Dynamic UX - Best Open Source References

For the click-to-move, undo, confirm-play UX you want:

| Repo | Stack | Key UX Features |
|------|-------|-----------------|
| [nielslange/react-backgammon](https://github.com/nielslange/react-backgammon) | React | **Undo/redo moves**, flip dice, save/load game, pip count display |
| [Shahbazarama/react-backgammon](https://github.com/Shahbazarama/react-backgammon) | React | **Click piece → click destination** (exactly what you described) |
| [timiles/backgammon](https://github.com/timiles/backgammon) | TypeScript/HTML/CSS | Clean pure-web implementation, human vs CPU, no framework bloat |
| [quasoft/backgammonjs](https://github.com/quasoft/backgammonjs) | JS | Multiplayer, mobile-friendly, extensible rule variants |
| [sam-swarr/backgammon](https://github.com/sam-swarr/backgammon) | React + Firestore | Online multiplayer with private lobbies |

**Best starting point**: `Shahbazarama/react-backgammon` for the click-to-move pattern, `nielslange/react-backgammon` for undo/redo architecture.

---

## Recommended Architecture

```
┌─────────────────────────────────────────┐
│           Your Modern UI                │
│  (React/Svelte - click moves, undo)     │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Your MCTS Engine                │
│  (This is where your work goes)         │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│            gnubg                        │
│  - Legal moves    - Position eval       │
│  - Board state    - Benchmark AI        │
└─────────────────────────────────────────┘
```

---

## What You Actually Need to Build

1. **MCTS algorithm** - The core of your project
2. **Game state wrapper** - Thin layer over gnubg for your MCTS to interact with
3. **Modern UX** - Copy patterns from the React repos above:
   - Click checker → highlight valid destinations → click to confirm
   - Undo button (store move history as stack)
   - "Confirm turn" button before dice pass to opponent

---

## Quick Start

```python
import gnubg

gnubg.initnet()

# Get legal moves for a position + dice roll
board = gnubg.board_from_position_id("4HPwATDgc/ABMA")
legal_moves = gnubg.moves(board, die1=3, die2=1)

# Your MCTS explores these moves
# gnubg handles all the rule validation
```

---

## Sources
- [gnubg-nn-pypi](https://github.com/reayd-falmouth/gnubg-nn-pypi)
- [nielslange/react-backgammon](https://github.com/nielslange/react-backgammon)
- [Shahbazarama/react-backgammon](https://github.com/Shahbazarama/react-backgammon)
- [timiles/backgammon](https://github.com/timiles/backgammon)