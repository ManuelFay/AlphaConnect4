class BaseAgent:
    def __init__(self):
        self.ai_confidence: float = 0.5

    def move(self, board, turn) -> int:
        raise NotImplementedError
