from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    app_name: str = "AdaptiveML DSS"
    app_env: str = "dev"
    random_state: int = 42
    storage_root: Path = Field(default=Path("storage"))

    model_config = SettingsConfigDict(env_prefix="ADAPTIVEML_", extra="ignore")

    @property
    def datasets_dir(self) -> Path:
        """Return the directory that stores dataset snapshots."""
        return self.storage_root / "datasets"

    @property
    def artifacts_dir(self) -> Path:
        """Return the directory that stores model and report artifacts."""
        return self.storage_root / "artifacts"

    @property
    def registry_dir(self) -> Path:
        """Return the directory that stores registry metadata files."""
        return self.storage_root / "registry"

    def ensure_directories(self) -> None:
        """Create all required storage directories if they are missing."""
        for path in (
            self.storage_root,
            self.datasets_dir,
            self.artifacts_dir,
            self.registry_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance with initialized directories."""
    settings = Settings()
    settings.ensure_directories()
    return settings
