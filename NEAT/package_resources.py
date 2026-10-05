"""Package-owned runtime resources exposed through Traversable objects."""

from importlib.resources import files
from importlib.resources.abc import Traversable


def assistant_knowledge_root() -> Traversable:
    """Return the package resource directory containing approved assistant knowledge."""

    return files("NEAT.knowledge")


def launch_splash_resource() -> Traversable:
    """Return the installed package's launch splash resource."""

    return files("NEAT").joinpath("assets", "launch_splash.png")
