"""Learners supported by the WALL experiment entry point."""
from .q_learner import QLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner

REGISTRY = {
    "q_learner": QLearner,
    "dmaq_qatten_learner": DMAQ_qattenLearner,
}
