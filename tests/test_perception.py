import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from config import GameConfig, AgentConfig, RenderConfig
from game.engine import GameEngine
from game.renderer import GameRenderer
from agent.perception import VLMPerception


def test_rule_based_perception():
    config = GameConfig(seed=42)
    engine = GameEngine(config)
    state = engine.reset()

    renderer = GameRenderer(RenderConfig())
    screenshot = renderer.render(state)

    perception = VLMPerception(AgentConfig(), offline=True)
    description = perception.perceive(screenshot, state=state)

    assert "player" in description.lower()
    assert "goal" in description.lower()
    assert len(description) > 50
    print(f"✓ Perception output ({len(description)} chars):")
    print(f"  {description[:200]}...")
    print("✓ test_rule_based_perception passed")


def test_renderer_produces_image():
    config = GameConfig(seed=42)
    engine = GameEngine(config)
    state = engine.reset()

    renderer = GameRenderer(RenderConfig())
    img = renderer.render(state)

    assert isinstance(img, Image.Image)
    assert img.width > 0
    assert img.height > 0
    print(f"✓ Rendered image: {img.width}x{img.height}")
    print("✓ test_renderer_produces_image passed")


if __name__ == "__main__":
    test_rule_based_perception()
    test_renderer_produces_image()
    print("\n✅ All perception tests passed!")
