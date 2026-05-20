from __future__ import annotations

import os


_PUBLIC_DEMO_ENV = "REG_MONKEY_PUBLIC_DEMO_MODE"
_TRUTHY_VALUES = {"1", "true", "yes", "on"}


def is_public_demo_mode() -> bool:
    """Return whether public-demo mode is explicitly enabled."""

    return os.environ.get(_PUBLIC_DEMO_ENV, "").strip().lower() in _TRUTHY_VALUES
