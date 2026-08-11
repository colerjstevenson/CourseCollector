"""Pipeline stage entrypoints."""

from .collect import run_collect
from .scrape import run_scrape
from .combine import run_combine
from .match import run_match

__all__ = ["run_collect", "run_scrape", "run_combine", "run_match"]
