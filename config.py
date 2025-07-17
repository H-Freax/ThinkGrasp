# Camera configuration for ThinkGrasp
# Update these parameters to match your camera's intrinsic parameters

class CameraConfig:
    """Camera configuration parameters"""
    
    # Camera intrinsic parameters - UPDATE THESE FOR YOUR CAMERA
    WIDTH = 640
    HEIGHT = 480
    FX = 382.8567  # Focal length x
    FY = 382.4391  # Focal length y
    CX = 331.3490  # Principal point x
    CY = 247.1126  # Principal point y
    SCALE = 1000.0  # Depth scale factor
    
    # Image processing region - UPDATE FOR YOUR CAMERA RESOLUTION
    XMIN = 0
    YMIN = 0
    XMAX = 480
    YMAX = 640
    
    @classmethod
    def get_camera_info(cls):
        """Returns camera parameters in CameraInfo format"""
        from models.FGC_graspnet.utils.data_utils import CameraInfo
        return CameraInfo(
            width=cls.WIDTH,
            height=cls.HEIGHT,
            fx=cls.FX,
            fy=cls.FY,
            cx=cls.CX,
            cy=cls.CY,
            scale=cls.SCALE
        )
    
    @classmethod
    def get_processing_region(cls):
        """Returns image processing region coordinates"""
        return cls.XMIN, cls.YMIN, cls.XMAX, cls.YMAX