from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AdaptiveML DSS"
    app_env: str = "dev"
    random_state: int = 42
    storage_root: Path = Field(default=Path("storage"))

    model_config = SettingsConfigDict(env_prefix="ADAPTIVEML_", extra="ignore")

    @property
    def datasets_dir(self) -> Path:
        return self.storage_root / "datasets"

    @property
    def artifacts_dir(self) -> Path:
        return self.storage_root / "artifacts"

    @property
    def registry_dir(self) -> Path:
        return self.storage_root / "registry"

    def ensure_directories(self) -> None:
        for path in (
            self.storage_root,
            self.datasets_dir,
            self.artifacts_dir,
            self.registry_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
