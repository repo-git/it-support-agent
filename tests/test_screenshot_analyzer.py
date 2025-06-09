import json
import os
import sys
import types
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub ModelManager to avoid importing heavy dependencies
stub = types.ModuleType("model_manager")
class DummyModelManager:
    pass
stub.ModelManager = DummyModelManager
sys.modules["src.ai.model_manager"] = stub

from src.vision.screenshot_analyzer import ScreenshotAnalyzer


def _new_analyzer():
    # Bypass __init__ to avoid heavy dependencies
    return ScreenshotAnalyzer.__new__(ScreenshotAnalyzer)


def test_fallback_analysis_no_red():
    analyzer = _new_analyzer()
    img = Image.new("RGB", (100, 100), color="white")
    result = json.loads(analyzer._fallback_analysis(img))

    assert result["system_info"]["screen_resolution"] == "100x100"
    assert result["severity"] == "low"
    assert result["issues"] == []


def test_fallback_analysis_with_red():
    analyzer = _new_analyzer()
    img = Image.new("RGB", (100, 100), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 20, 20], fill="red")
    result = json.loads(analyzer._fallback_analysis(img))

    assert result["severity"] == "medium"
    assert len(result["issues"]) == 1
    assert result["issues"][0]["type"] == "possible_error"


def test_parse_analysis_result_json():
    analyzer = _new_analyzer()
    json_text = '{"issues": [{"description": "ok", "type": "info"}]}'
    parsed = analyzer._parse_analysis_result(json_text)

    assert parsed["issues"][0]["description"] == "ok"


def test_parse_analysis_result_text():
    analyzer = _new_analyzer()
    text = "Generic failure"
    parsed = analyzer._parse_analysis_result(text)

    assert parsed["issues"][0]["description"] == text
    assert parsed["system_info"].get("raw_analysis")
