from alphaconnect4.interfaces.backgammon_board import BackgammonBoard
from alphaconnect4.interfaces.backgammon_mcts_interface import BackgammonTree


def test_starting_positions():
    board = BackgammonBoard()
    assert board.points[23] == 2
    assert board.points[12] == 5
    assert board.points[7] == 3
    assert board.points[5] == 5
    assert board.points[0] == -2
    assert board.points[11] == -5
    assert board.points[16] == -3
    assert board.points[18] == -5


def test_bar_entry_allows_open_point():
    board = BackgammonBoard()
    board.points[:] = 0
    board.turn = 0
    board.bar[0] = 1
    board.points[23] = -2
    board.points[22] = 0

    actions = board.get_legal_actions((1, 2))
    assert any(any(move[0] == "bar" and move[1] == 22 for move in action) for action in actions)


def test_bear_off_when_all_home():
    board = BackgammonBoard()
    board.points[:] = 0
    board.turn = 0
    board.bar[0] = 0
    board.points[0] = 1

    actions = board.get_legal_actions((2, 1))
    assert any(any(move[1] == "off" for move in action) for action in actions)


def test_apply_action_hits_blot():
    board = BackgammonBoard()
    board.points[:] = 0
    board.turn = 0
    board.points[5] = 1
    board.points[3] = -1

    board.apply_action([(5, 3, 2)])
    assert board.points[3] == 1
    assert board.bar[1] == 1


def test_tree_reward_matches_turn():
    board = BackgammonBoard()
    board.borne_off[0] = 15
    board.turn = 0
    tree = BackgammonTree(board, (1, 1))
    assert tree.reward() == 1.0
