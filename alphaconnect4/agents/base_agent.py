class BaseAgent:
    def __init__(self):
        self.ai_confidence: float = 0.5

    def move(self, board, turn) -> int:
        raise NotImplementedError

    def kill_agent(self, result):
        """Results: 1 - Win, 0.5 - Tie, 0 - Loss"""
        print(f"{self.__class__.__name__} has been killed with result {result}")
