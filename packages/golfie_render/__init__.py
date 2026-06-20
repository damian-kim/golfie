"""golfie_render

Bridges Python-side ShotResult/Session data into the JSON payload shape
the browser-based Three.js driving range (apps/web/frontend) expects.
There is no 3D rendering code in Python -- that all lives in the
frontend -- this package only defines and produces the wire format.
"""

__all__ = ["threejs"]
