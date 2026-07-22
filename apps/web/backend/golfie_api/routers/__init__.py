from golfie_api.routers.sessions import router as sessions_router
from golfie_api.routers.calibration import router as calibration_router
from golfie_api.routers.video_trimmer import router as video_trimmer_router

__all__ = ["sessions_router", "calibration_router", "video_trimmer_router"]
