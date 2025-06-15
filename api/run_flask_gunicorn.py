from flask import Flask

from api.run_flask import FlaskConfig, RunFlaskCommand


def create_app() -> Flask:
    # TODO: Add config files
    """Create the Flask app object and return it (without starting the app).
    Entry point for gunicorn
    """
    from alphaconnect4.agents.mcts_agent import MCTSAgent

    command = RunFlaskCommand(FlaskConfig(), MCTSAgent())

    return command.run(start=False)
