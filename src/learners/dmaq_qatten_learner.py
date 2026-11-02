"""QPLEX uses the shared, fully evaluated WALL learning pipeline."""
from .q_learner import QLearner


class DMAQ_qattenLearner(QLearner):
    def __init__(self, mac, scheme, logger, args, qdifference_transformer, planning_transformer):
        if args.mixer != "dmaq":
            raise ValueError("Optimized WALL QPLEX uses mixer='dmaq'; the optional Qatten variant is unsupported.")
        super().__init__(mac, scheme, logger, args, qdifference_transformer, planning_transformer)
