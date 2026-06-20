"""golfie_core

Shared schemas, coordinate conventions, configuration, and metric types
used by every other Golfie package (golfie_cv, golfie_physics,
golfie_render) and by the FastAPI backend.

Nothing in this package should depend on OpenCV, SciPy, or any other
heavy CV/physics dependency -- it is meant to be the lightest, most
stable layer in the system.
"""

__all__ = ["schemas", "coordinates", "config", "metrics"]
