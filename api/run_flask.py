import json
import logging
import os
import uuid
from dataclasses import dataclass

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from alphaconnect4.agents.base_agent import BaseAgent
from alphaconnect4.constants.constants import COLUMN_COUNT, ROW_COUNT


@dataclass
class FlaskConfig:
    port: int = None
    host: str = None
    debug: bool = None
    threaded: bool = True


class RunFlaskCommand:
    logger = logging.getLogger(__name__)

    def __init__(self, flask_config: FlaskConfig, agent: BaseAgent, log_dir=None):
        self.agent = agent
        self.flask_config = flask_config
        self._log_dir = log_dir

    @staticmethod
    def board_from_string(str_board: str) -> np.ndarray:
        board = np.fromstring(str_board, dtype=int, sep=" ")
        return np.reshape(board, (ROW_COUNT, COLUMN_COUNT))

    def get_move(self):
        """Routine that runs the QA inference pipeline."""
        user_request = request.get_json()
        current_board = user_request.get("board") or ""
        current_turn = user_request.get("turn") or ""

        board = self.board_from_string(current_board)

        result = int(self.agent.move(board, turn=int(current_turn)))
        confidence = float(self.agent.ai_confidence)

        if self._log_dir is not None:
            with open(os.path.join(self._log_dir, f"{uuid.uuid4()}.json"), "w", encoding="utf8") as output_file:
                json.dump({"optimal_move": result, "confidence": confidence}, output_file, ensure_ascii=False)

        return jsonify(optimal_move=result, confidence=confidence)

    def run(self, start=True):
        app = Flask(__name__, static_folder=None)
        CORS(app)

        if self._log_dir is not None:
            os.makedirs(self._log_dir, exist_ok=True)

        app.add_url_rule("/get_move", "get_move", self.get_move, methods=["POST"])

        if start:
            app.run(
                port=self.flask_config.port,
                host=self.flask_config.host,
                debug=self.flask_config.debug,
                threaded=self.flask_config.threaded,
            )
        return app


if __name__ == "__main__":
    from alphaconnect4.agents.mcts_agent import MCTSAgent

    command = RunFlaskCommand(FlaskConfig(), MCTSAgent())
    command.run()
