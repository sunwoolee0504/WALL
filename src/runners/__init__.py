"""The supported WALL collector (legacy parallel runner is archived)."""
from .episode_runner import EpisodeRunner

REGISTRY = {"episode": EpisodeRunner}
