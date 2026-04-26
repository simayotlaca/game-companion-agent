"""VLM perception module for game state understanding.

Provides multiple backends for converting game screenshots into structured
text descriptions: local BLIP-2, API-based, and rule-based (offline).
"""

import logging
from typing import Optional, Protocol, Union

from PIL import Image

from config import AgentConfig
from game.engine import GameState

logger = logging.getLogger(__name__)

PERCEPTION_PROMPT = """Describe what you see in this game screenshot.
The game is a tactical grid game. Identify:
1. Player position and health
2. Enemy positions, health, and proximity to player
3. Health pack locations
4. Goal location
5. Immediate threats or opportunities

Be concise and structured. Use coordinates (x,y) when possible."""


class PerceptionModel(Protocol):
    """Protocol for perception backend implementations."""

    def describe(self, image: Image.Image) -> str: ...


class BLIP2Perception:
    """Local BLIP-2 model backend for visual perception.

    Loads a BLIP-2 model via HuggingFace Transformers and runs
    inference locally on GPU or CPU.

    Args:
        config: Agent configuration with model name and device settings.

    Raises:
        RuntimeError: If model loading fails (e.g. insufficient VRAM).
    """

    def __init__(self, config: AgentConfig) -> None:
        import torch
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        self.device = self._resolve_device(config.device)
        dtype_map = {"float16": torch.float16, "float32": torch.float32,
                     "bfloat16": torch.bfloat16}
        self.dtype = dtype_map.get(config.vlm_dtype, torch.float16)
        self.max_tokens = config.vlm_max_new_tokens

        logger.info("Loading VLM %s on %s...", config.vlm_model_name, self.device)
        try:
            self.processor = Blip2Processor.from_pretrained(config.vlm_model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                config.vlm_model_name,
                torch_dtype=self.dtype,
                device_map=self.device if self.device != "cpu" else None,
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            logger.info("VLM loaded successfully.")
        except Exception as e:
            logger.error("Failed to load VLM: %s", e)
            raise RuntimeError(f"VLM initialization failed: {e}") from e

    def describe(self, image: Image.Image) -> str:
        """Generate a text description of the game screenshot.

        Args:
            image: PIL Image of the rendered game board.

        Returns:
            Structured text description of the game state.
        """
        inputs = self.processor(
            images=image,
            text=PERCEPTION_PROMPT,
            return_tensors="pt"
        ).to(self.device, self.dtype)

        with __import__("torch").no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
        description = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return description.strip()

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' device to 'cuda' or 'cpu'."""
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device


class APIPerception:
    """API-based perception backend using OpenAI-compatible endpoints.

    Sends game screenshots as base64-encoded images to a remote
    VLM endpoint for description generation.

    Args:
        config: Agent configuration with API URL and token settings.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.api_url = config.vlm_api_url
        self.max_tokens = config.vlm_max_new_tokens

    def describe(self, image: Image.Image) -> str:
        """Send screenshot to API and return the text description.

        Args:
            image: PIL Image of the rendered game board.

        Returns:
            Structured text description from the API.

        Raises:
            ConnectionError: If the API endpoint is unreachable.
            ValueError: If the API response format is unexpected.
        """
        import requests
        import base64
        from io import BytesIO

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        payload = {
            "model": "default",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PERCEPTION_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }],
            "max_tokens": self.max_tokens,
        }

        try:
            resp = requests.post(self.api_url, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.ConnectionError as e:
            logger.error("VLM API unreachable at %s: %s", self.api_url, e)
            raise ConnectionError(f"VLM API unreachable: {self.api_url}") from e
        except (KeyError, IndexError) as e:
            logger.error("Unexpected VLM API response format: %s", e)
            raise ValueError("Unexpected VLM API response format") from e


class RuleBasedPerception:
    """Offline rule-based perception backend (no model needed).

    Generates structured text descriptions directly from game state data,
    bypassing visual perception entirely. Used for offline demos and testing.
    """

    def __init__(self) -> None:
        self._last_state: Optional[GameState] = None

    def set_state(self, state: GameState) -> None:
        """Inject the current game state for rule-based description."""
        self._last_state = state

    def describe(self, image: Image.Image) -> str:
        """Generate a text description from the injected game state.

        Args:
            image: Ignored in rule-based mode (state is used directly).

        Returns:
            Structured text description of entity positions and threats.
        """
        if self._last_state is None:
            return "Unable to perceive game state."

        state = self._last_state
        p = state.player
        lines = [
            f"The player (blue) is at position ({p.x},{p.y}) with {p.hp}/{p.max_hp} HP.",
        ]

        for enemy in state.living_enemies:
            dist = p.distance_to(enemy)
            threat = "within attack range" if dist <= p.attack_range else f"{dist} tiles away"
            dx, dy = enemy.x - p.x, enemy.y - p.y
            dirs = []
            if dy < 0: dirs.append("north")
            if dy > 0: dirs.append("south")
            if dx > 0: dirs.append("east")
            if dx < 0: dirs.append("west")
            dir_str = "-".join(dirs) if dirs else "same position"
            lines.append(
                f"Enemy #{enemy.entity_id} (red) is at ({enemy.x},{enemy.y}), "
                f"{threat}, to the {dir_str}, with {enemy.hp}/{enemy.max_hp} HP."
            )

        for hp in state.available_health_packs:
            dist = p.distance_to(hp)
            lines.append(f"A health pack (+{hp.restore_amount} HP) is at ({hp.x},{hp.y}), {dist} tiles away.")

        goal_dist = p.distance_to(state.goal)
        lines.append(f"The goal (gold star) is at ({state.goal.x},{state.goal.y}), {goal_dist} tiles away.")

        if state.living_enemies:
            closest = min(state.living_enemies, key=lambda e: p.distance_to(e))
            if p.distance_to(closest) <= p.attack_range:
                lines.append(f"THREAT: Enemy #{closest.entity_id} is in attack range!")
        if p.hp < p.max_hp * 0.4:
            lines.append("WARNING: Player health is critically low.")

        return " ".join(lines)


class VLMPerception:
    """Unified perception interface that dispatches to the appropriate backend.

    Selects between BLIP-2 (local), API, or rule-based perception
    based on configuration and offline mode flag.

    Args:
        config: Agent configuration with model and API settings.
        offline: If True, use rule-based backend (no GPU/API needed).
    """

    def __init__(self, config: AgentConfig, offline: bool = False) -> None:
        if offline:
            self._backend: Union[RuleBasedPerception, APIPerception, BLIP2Perception] = RuleBasedPerception()
        elif config.use_api:
            self._backend = APIPerception(config)
        else:
            self._backend = BLIP2Perception(config)
        self._is_rule_based = offline

    def perceive(self, image: Image.Image, state: Optional[GameState] = None) -> str:
        """Perceive the game state from a screenshot.

        Args:
            image: Rendered game board image.
            state: Optional GameState for rule-based backend injection.

        Returns:
            Structured text description of the perceived game state.
        """
        if self._is_rule_based and state is not None:
            self._backend.set_state(state)
        return self._backend.describe(image)
