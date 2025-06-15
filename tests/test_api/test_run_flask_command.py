# import json
import os
from shutil import rmtree
from unittest import TestCase
from unittest.mock import create_autospec  # patch

from flask import Flask

from alphaconnect4.agents.base_agent import BaseAgent
from api.run_flask import FlaskConfig, RunFlaskCommand


class TestRunFlaskCommand(TestCase):
    def setUp(self) -> None:
        self.flask_config = FlaskConfig()
        self.retriever = create_autospec(BaseAgent, spec_set=False)
        self.log_dir = os.path.join(os.path.dirname(__file__), "logs")
        self.command = RunFlaskCommand(
            self.flask_config,
            self.retriever,
            self.log_dir,
        )

    def tearDown(self) -> None:
        if os.path.exists(self.log_dir):
            rmtree(self.log_dir)

    def test_run(self):
        app = self.command.run(start=False)
        self.assertIsInstance(app, Flask)
        routes = {rule.rule for rule in app.url_map.iter_rules()}
        self.assertEqual({"/get_move"}, routes)
        self.assertTrue(os.path.exists(self.log_dir))

    def test_board_from_str(self):
        board = self.command.board_from_string("0 " * (36) + "1 2 1 1 2 2 ")
        self.assertEqual(board.shape, (6, 7))
