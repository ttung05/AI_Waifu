"""
AI Waifu Source Package
Consolidated and optimized AI Waifu modules
"""

from .system_detector import SystemDetector
from .unified_setup_manager import UnifiedSetupManager
from .ai_waifu_core import AIWaifuCore, DeviceManager
from .web_interface import AIWaifuWebInterface

__version__ = "2.0.0"
__author__ = "AI Waifu Team"

# Export main classes
__all__ = [
    'SystemDetector',
    'UnifiedSetupManager', 
    'AIWaifuCore',
    'DeviceManager',
    'AIWaifuWebInterface'
]