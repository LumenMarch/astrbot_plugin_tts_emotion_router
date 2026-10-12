"""Allow tests to run from a checkout with any directory name."""
import sys
import types
from pathlib import Path

package = types.ModuleType("astrbot_plugin_tts_emotion_router")
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package

# Avoid importing a locally installed AstrBot and migrating its runtime config.
astrbot = types.ModuleType("astrbot")
astrbot.__path__ = []
sys.modules["astrbot"] = astrbot
