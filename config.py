import os
from dataclasses import dataclass, field
from typing import Optional


MODEL_REGISTRY = {
    "blip2": "Salesforce/blip2-opt-2.7b",
    "blip2-large": "Salesforce/blip2-opt-6.7b",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral-7b-v3": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
}

DEFAULT_VLM = MODEL_REGISTRY["blip2"]
DEFAULT_LLM = MODEL_REGISTRY["mistral-7b"]


@dataclass
class GameConfig:
    grid_width: int = 8
    grid_height: int = 8
    max_turns: int = 50
    num_enemies: int = 3
    num_health_packs: int = 2
    player_hp: int = 100
    player_attack: int = 25
    player_attack_range: int = 2
    player_move_points: int = 1
    enemy_hp: int = 100
    enemy_attack: int = 15
    enemy_detection_range: int = 3
    health_pack_restore: int = 30
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        assert self.grid_width >= 4, f"grid_width must be >= 4, got {self.grid_width}"
        assert self.grid_height >= 4, f"grid_height must be >= 4, got {self.grid_height}"
        assert self.max_turns >= 1, f"max_turns must be >= 1, got {self.max_turns}"
        assert self.num_enemies >= 0, f"num_enemies must be >= 0, got {self.num_enemies}"
        assert self.num_health_packs >= 0, f"num_health_packs must be >= 0, got {self.num_health_packs}"
        assert self.player_hp > 0, f"player_hp must be > 0, got {self.player_hp}"
        assert self.player_attack > 0, f"player_attack must be > 0, got {self.player_attack}"
        assert self.player_attack_range >= 1, f"player_attack_range must be >= 1, got {self.player_attack_range}"
        assert self.player_move_points >= 1, f"player_move_points must be >= 1, got {self.player_move_points}"
        assert self.enemy_hp > 0, f"enemy_hp must be > 0, got {self.enemy_hp}"
        assert self.health_pack_restore > 0, f"health_pack_restore must be > 0, got {self.health_pack_restore}"
        max_entities = self.num_enemies + self.num_health_packs + 2  # +2: player + goal
        assert max_entities <= self.grid_width * self.grid_height, (
            f"Too many entities ({max_entities}) for grid size {self.grid_width}x{self.grid_height}"
        )


@dataclass
class AgentConfig:
    vlm_model_name: str = field(default_factory=lambda: os.getenv("VLM_MODEL", DEFAULT_VLM))
    llm_model_name: str = field(default_factory=lambda: os.getenv("LLM_MODEL", DEFAULT_LLM))

    use_api: bool = False
    vlm_api_url: str = "http://localhost:8000/v1/chat/completions"
    llm_api_url: str = "http://localhost:8001/v1/chat/completions"

    vlm_max_new_tokens: int = 200
    llm_max_new_tokens: int = 300
    llm_temperature: float = 0.3
    llm_top_p: float = 0.9

    device: str = "auto"
    vlm_dtype: str = "float16"
    llm_dtype: str = "float16"
    load_in_4bit: bool = False

    max_action_retries: int = 2
    fallback_to_heuristic: bool = True

    # Reproducibility
    seed: Optional[int] = None
    deterministic_mode: bool = False  # If True, sets torch/numpy global seeds + cudnn flags

    def __post_init__(self) -> None:
        assert self.vlm_model_name, "vlm_model_name cannot be empty"
        assert self.llm_model_name, "llm_model_name cannot be empty"
        assert self.vlm_max_new_tokens >= 10, f"vlm_max_new_tokens must be >= 10, got {self.vlm_max_new_tokens}"
        assert self.llm_max_new_tokens >= 10, f"llm_max_new_tokens must be >= 10, got {self.llm_max_new_tokens}"
        assert 0.0 <= self.llm_temperature <= 2.0, f"llm_temperature must be in [0, 2], got {self.llm_temperature}"
        assert 0.0 < self.llm_top_p <= 1.0, f"llm_top_p must be in (0, 1], got {self.llm_top_p}"
        assert self.max_action_retries >= 0, f"max_action_retries must be >= 0, got {self.max_action_retries}"
        assert self.vlm_dtype in ("float16", "float32", "bfloat16"), f"invalid vlm_dtype: {self.vlm_dtype}"
        assert self.llm_dtype in ("float16", "float32", "bfloat16"), f"invalid llm_dtype: {self.llm_dtype}"


@dataclass
class RenderConfig:
    cell_size: int = 64
    padding: int = 2
    font_size: int = 14
    bg_color: tuple = (20, 20, 30)
    grid_color: tuple = (40, 40, 55)
    player_color: tuple = (50, 130, 240)
    enemy_color: tuple = (220, 50, 50)
    health_pack_color: tuple = (50, 200, 80)
    goal_color: tuple = (240, 200, 50)
    empty_color: tuple = (30, 30, 42)
    text_color: tuple = (220, 220, 220)


@dataclass
class AppConfig:
    game: GameConfig = field(default_factory=GameConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    verbose: bool = True
    save_screenshots: bool = False
    screenshot_dir: str = "screenshots"
