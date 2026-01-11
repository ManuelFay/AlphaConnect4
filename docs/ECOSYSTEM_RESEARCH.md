# Backgammon Ecosystem Research Summary

## Overview

Deep dive into existing Python/open-source backgammon tools to inform our project direction.

**Total GitHub repositories found**: 105+ public projects tagged with backgammon

---

## Tier 1: Production-Ready Engines (Use These)

### 1. gnubg-nn-pypi (RECOMMENDED)
**[GitHub](https://github.com/reayd-falmouth/gnubg-nn-pypi)** | **[PyPI](https://pypi.org/project/gnubg/)** | **[Docs](https://gnubg.readthedocs.io/)**

- **What**: Python 3 bindings to GNU Backgammon's neural net engine
- **License**: GPL-3.0
- **Install**: `pip install gnubg`

**Key APIs**:
```python
gnubg.initnet()                           # Initialize engine
gnubg.board_from_position_id(id)          # Decode position ID -> board
gnubg.position_id(board)                  # Encode board -> position ID
gnubg.classify(board)                     # Basic classification (6 types)
gnubg.probabilities(board, ply)           # Win/gammon probabilities
gnubg.best_move(board, d1, d2, n=2)       # Best move at n-ply
gnubg.moves(board, d1, d2)                # Legal moves
gnubg.rollout(board, ngames=1296)         # Monte Carlo rollout
equities.value(x_away, o_away)            # Match equity table
```

**Classification types**: `c_contact`, `c_race`, `c_crashed`, `c_bearoff`, `c_over`, `c_backcontain`

**Verdict**: This is our foundation. Position ID handling, neural net eval, basic classification - all for free.

---

### 2. wildbg
**[GitHub](https://github.com/carsten-wenderdel/wildbg)**

- **What**: Modern backgammon engine in Rust with HTTP/JSON API
- **License**: MIT/Apache 2.0 (permissive!)
- **Strength**: ~5.9 error rate on 1-pointers (competitive with GnuBG)

**Key features**:
- HTTP JSON API (Swagger UI)
- Cubeless and cubeful evaluation
- Trained neural networks available
- Can run as bot on backgammon servers

**Verdict**: Interesting alternative if we need permissive licensing. HTTP API could be useful for microservice architecture. But gnubg-nn-pypi is more mature.

---

### 3. pgx (sotetsuk) - 573 stars
**[GitHub](https://github.com/sotetsuk/pgx)**

- **What**: JAX-native GPU-accelerated game simulators for RL research
- **Games**: 25+ games including backgammon, chess, Go, shogi
- **Features**: JIT-compilable, vectorizable, SVG visualization
- **License**: Apache 2.0

**Verdict**: Interesting for RL research but overkill for our use case.

---

## Tier 2: Training/Study Tools (Learn From These)

### 3. AnkiGammon
**[Website](https://ankigammon.com/)** | Free & open source

- **What**: Converts backgammon analysis into Anki flashcards
- **Formats**: XGID, OGID, GNUID, .xg, .mat, .sgf files
- **Features**:
  - Drag-and-drop file import
  - Auto-runs GnuBG for analysis if needed
  - 6 board color themes
  - Move/cube decision cards
  - Error threshold filtering

**Verdict**: Great UX reference for training tools. Shows what players want: position visualization, spaced repetition, error filtering.

---

### 4. xgid2anki
**[GitHub](https://github.com/ngvlamis/xgid2anki)** | **[PyPI](https://pypi.org/project/xgid2anki/)**

- **What**: CLI tool to convert XGIDs to Anki decks
- **Stack**: Python, GnuBG, Playwright (headless Chrome), genanki
- **Uses**: bglog for board rendering

**Verdict**: Shows the pipeline: position -> analysis -> visualization -> flashcard. We could adapt this approach.

---

### 5. BGTrain
**[GitHub](https://github.com/gtzampanakis/bgtrain)** | **[Live](https://www.bgtrain.com/)**

- **What**: Web app for position quizzes
- **Stack**: Python backend, JavaScript frontend
- **Features**: ELO rating system, position database, quickfire quizzes

**Verdict**: Similar concept to our training tools. Worth studying their quiz UX.

---

### 6. BlunderDB
**[FFBG](https://www.ffbg.fr/actualites/536-blunderdb-un-nouvel-outil-gratuit-pour-progresser-)** | Standalone desktop software

- **What**: Position database software for tracking/studying blunders
- **Features**:
  - Aggregate positions from online/tournament play
  - Filter by various criteria
  - Tag and annotate positions
  - Create reference position catalogs
- **Format**: .db files

**Verdict**: Similar concept to our project. Could import their format later. Study their filtering/tagging UX.

---

## Tier 3: Board Visualization Libraries

### 7. bgboard (gtzampanakis)
**[GitHub](https://github.com/gtzampanakis/bgboard)** | MIT License

- **What**: Pure JS library for rendering backgammon boards (display only, no gameplay)
- **Features**:
  - No dependencies
  - Supports GNU Backgammon position format
  - Multiple boards per page
  - Scalable sizing
  - Shows checker moves, roll/double, take/drop decisions
- **Used by**: bgtrain.com

**Verdict**: Could adapt for our UI, or use as reference for Svelte component.

### 8. React Implementations

| Repo | Stars | Notes |
|------|-------|-------|
| [sam-swarr/backgammon](https://github.com/sam-swarr/backgammon) | - | React + Firestore, multiplayer |
| [AsadpourMohammad/Backgammon-React](https://github.com/AsadpourMohammad/Backgammon-React) | - | Playable on GitHub Pages |
| [ser-ge/backgammon](https://github.com/ser-ge/backgammon) | - | Flask + React + Socket.io |
| [lkowalick/backgammon-react](https://github.com/lkowalick/backgammon-react) | - | React + SVG rendering |

**Verdict**: Reference implementations for board rendering in React/JS.

### 9. Canvas/HTML5 Implementations

| Repo | Notes |
|------|-------|
| [Tuscann/CanvasGame](https://github.com/Tuscann/CanvasGame) | HTML5 Canvas + JS |
| [wbillingsley/play-backgammon](https://github.com/wbillingsley/play-backgammon) | Scala + Play + d3.js |
| [quasoft/backgammonjs](https://github.com/quasoft/backgammonjs) | JS, 98 stars, multiplayer |

---

## Tier 4: OpenAI Gym / RL Environments

### 10. gym-backgammon (dellalibera) - 53 stars
**[GitHub](https://github.com/dellalibera/gym-backgammon)**

- **What**: Backgammon as OpenAI Gym environment
- **Encoding**: Tesauro-style state representation
- **Features**: Legal move generation, proper dice handling

**Used by**: Multiple TD-Gammon implementations for training

### 11. Related RL Projects

| Repo | Framework | Notes |
|------|-----------|-------|
| [MateiCosa/backgammon-ai](https://github.com/MateiCosa/backgammon-ai) | PyTorch | Has GUI, uses gym env |
| [ardabbour/amca](https://github.com/ardabbour/amca) | - | DQN, PPO, SAC comparison |
| [natkinson1/backgammon-AI](https://github.com/natkinson1/backgammon-AI) | - | 350k self-play games |

---

## Tier 5: Statistics & Analysis Tools

### 12. lassehjorthmadsen/backgammon (R Package)
**[GitHub](https://rdrr.io/github/lassehjorthmadsen/backgammon/)**

- **What**: R package for backgammon analysis
- **Data**: 39,690 positions from 343 matches (Backgammon Galaxy)
- **Features**: Match winning chances, outcome probabilities, take points
- **Format**: Analyzed by GnuBG 4-ply

**Verdict**: Good reference for what statistics players want.

### 13. akiyoko/bgstat
**[GitHub](https://github.com/akiyoko/bgstat)**

- **What**: Statistics for Backgammon
- **Status**: Small project (0 stars)

---

## Tier 6: AI/Research Projects (Reference)

### 14. TD-Gammon Implementations

Multiple Python implementations of Tesauro's TD-Gammon:

| Repo | Framework | Notes |
|------|-----------|-------|
| [dellalibera/td-gammon](https://github.com/dellalibera/td-gammon) | PyTorch | Tests against GnuBG |
| [fomorians/td-gammon](https://github.com/fomorians/td-gammon) | TensorFlow | Eligibility traces |
| [MateiCosa/backgammon-ai](https://github.com/MateiCosa/backgammon-ai) | Python 3.11 | TD-learning |
| [jacobhilton/backgammon](https://github.com/jacobhilton/backgammon) | OCaml | 42-43% vs GnuBG |
| [nurialozano/nlgammon](https://github.com/nurialozano/nlgammon) | Python | Has GUI |

**Verdict**: Academic interest. We don't need to train our own NN - GnuBG already has world-class nets.

---

## Position ID Formats

### GNU Backgammon ID (GNUID)
- **Format**: 14-char base64 + optional match ID
- **Example**: `4HPwATDgc/ABMA:cYkAAAAAAAAA`
- **Encoding**: Bit string (1 per checker, 0 as separator) -> 10 bytes -> base64

### XGID (eXtreme Gammon ID)
- **Format**: 26-char position + metadata fields
- **Example**: `XGID=---BBBBAAA---Ac-bbccbAA-A-:1:1:-1:63:4:3:0:5:8`
- **Encoding**: A-Z = player 1 checkers, a-z = player 2, `-` = empty

### Conversion
- [bglab R package](https://lassehjorthmadsen.github.io/bglab/reference/gnuid2xgid.html) has gnuid2xgid conversion
- AnkiGammon supports all formats with auto-detection

**Verdict**: Use GNUID as primary (gnubg-nn-pypi native). Support XGID import for XG users.

---

## Commercial Software (Competitors)

| Software | Price | Strength | Notes |
|----------|-------|----------|-------|
| eXtreme Gammon (XG) | ~$60 | World #1 | Gold standard, XG2 powers Galaxy |
| Snowie | $100-380 | Strong | Multiple editions |
| JellyFish | ~$50 | Strong | Neural net based |
| Backgammon Galaxy | Free | XG2-powered | Online play + analysis |

**Verdict**: XG is the benchmark. We're not competing on engine strength (use GnuBG). Our value-add is fine-grained classification + training UX.

---

## Key Insights for Our Project

### What exists:
1. **Engine**: GnuBG neural net (use gnubg-nn-pypi)
2. **Position encoding**: GNUID/XGID standards
3. **Basic classification**: 6 types in GnuBG
4. **Training tools**: Anki-based flashcards (AnkiGammon, xgid2anki)
5. **Analysis**: XG/GnuBG do rollouts, equity, best move

### What's missing (our opportunity):
1. **Fine-grained classification**: 17 types vs 6
2. **Modern UI**: Svelte vs dated GnuBG/XG interfaces
3. **Position type training**: Drills focused on recognizing position types
4. **Strategic education**: GM commentary explaining the "why"
5. **Web-first experience**: No install, instant access

### Recommended architecture:
```
+-----------------------------------------------------+
|                    Our System                        |
+-----------------------------------------------------+
|  SvelteKit UI (beautiful, modern)                   |
|       |                                             |
|  FastAPI Backend                                    |
|       |                                             |
|  gnubg-nn-pypi (engine layer)                       |
|    - Position ID encode/decode                      |
|    - Neural net evaluation                          |
|    - Basic classification (c_contact, c_race, etc)  |
|       |                                             |
|  Our Classification Layer                           |
|    - 17-category fine-grained taxonomy              |
|    - Feature extraction                             |
|    - Strategic insights                             |
+-----------------------------------------------------+
```

---

## Next Steps

1. **Install gnubg-nn-pypi** and verify it works
2. **Adopt their board format** (2x25 array) as our internal representation
3. **Implement our classifier** on top of their `classify()` output
4. **Build Svelte UI** for board visualization
5. **Create training mode** with position type drills

---

## Sources

- [gnubg-nn-pypi GitHub](https://github.com/reayd-falmouth/gnubg-nn-pypi)
- [wildbg GitHub](https://github.com/carsten-wenderdel/wildbg)
- [AnkiGammon](https://ankigammon.com/)
- [xgid2anki GitHub](https://github.com/ngvlamis/xgid2anki)
- [BGTrain GitHub](https://github.com/gtzampanakis/bgtrain)
- [TD-Gammon implementations](https://github.com/topics/backgammon)
- [GNU Backgammon Manual](https://www.gnu.org/software/gnubg/manual/)
- [XG File Format](https://www.extremegammon.com/XGformat.aspx)
