# AlphaBackgammon: Design Notes & Roadmap

## Overview
This module adds a Backgammon implementation alongside the existing Connect4 stack. It includes:

- A rules engine (`BackgammonBoard`) with legal move generation, dice handling, bar/bearing-off logic.
- A pure MCTS agent (`BackgammonMCTSAgent`) leveraging the existing `MCTS` engine.
- A pygame UI (`BackgammonVisualEngine`) and gameplay loop (`BackgammonGame`).

## How the Rules Engine Works

- **Board Representation:** `points` is a length-24 array; positive counts are player 0, negative counts are player 1.
- **Bar/Borne Off:** `bar` counts captured checkers; `borne_off` counts checkers removed from the board.
- **Turn & Movement:** player 0 moves from high index to low index; player 1 moves from low to high.
- **Legal Actions:** dice are expanded into possible move sequences. If doubles are rolled, four moves are attempted.
- **Constraints:** when both dice are available, the engine keeps only the longest legal move sequences; if only one die
  is playable it prefers the higher die when applicable.

## MCTS Integration

`BackgammonTree` adapts `BackgammonBoard` into the generic MCTS `Node` interface:

- Each node includes the current dice roll.
- `find_children` expands all legal actions for the current dice; if none, it creates a pass node.
- `reward` returns a win/loss based on who has borne off all checkers.

## UI Controls

- **Arrow keys:** cycle through legal move sequences.
- **Enter:** confirm the currently selected move.
- **Close window:** exit.

## Roadmap

1. **Neural MCTS Integration**
   - Add a backgammon-specific neural interface and training pipeline.
   - Use the policy/value network to guide rollouts and improve move selection.

2. **Rule Coverage & Validation**
   - Add unit tests for bearing-off edge cases and bar-entry logic.
   - Validate move generation against standard backgammon scenarios.

3. **UI Improvements**
   - Highlight selected move path on the board.
   - Add dice visuals and per-move animation.
   - Display win state and match score.

4. **Performance/UX**
   - Cache legal move sequences per dice roll.
   - Add adjustable MCTS time controls in the UI.
