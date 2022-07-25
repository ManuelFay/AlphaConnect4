class NeuralInterface:
    def score(self, node):
        """Flip board so that agent is always with pieces #1
        Score is from the POV of the next to play"""
        raise NotImplementedError
