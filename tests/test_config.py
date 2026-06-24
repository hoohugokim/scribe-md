"""Tests for layered scribe-md configuration loading."""

import pytest

from scribe_md import config
from scribe_md.config import ScribeMdConfig


def test_project_config_overrides_user_config(tmp_path, monkeypatch):
    user_config = tmp_path / "user.toml"
    user_config.write_text(
        """
[defaults]
model = "user-model"
chunk_seconds = 120

[output]
directory = "user-output"
""",
        encoding="utf-8",
    )

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / config.PROJECT_CONFIG_NAME).write_text(
        """
[defaults]
model = "project-model"

[output]
directory = "project-output"
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "USER_CONFIG_PATH", user_config)
    monkeypatch.chdir(project_dir)

    cfg = config.load_config()

    assert cfg.model == "project-model"
    assert cfg.chunk_seconds == 120
    assert cfg.output_directory == "project-output"
    assert cfg._sources == [
        "built-in defaults",
        str(user_config),
        str(project_dir / config.PROJECT_CONFIG_NAME),
    ]


def test_find_project_config_walks_up_from_cwd(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    nested = project_dir / "a" / "b"
    nested.mkdir(parents=True)
    project_config = project_dir / config.PROJECT_CONFIG_NAME
    project_config.write_text("[defaults]\nmodel = \"project-model\"\n", encoding="utf-8")

    monkeypatch.chdir(nested)

    assert config._find_project_config() == project_config


def test_malformed_toml_is_ignored(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / config.PROJECT_CONFIG_NAME).write_text("[defaults\n", encoding="utf-8")

    monkeypatch.setattr(config, "USER_CONFIG_PATH", tmp_path / "missing.toml")
    monkeypatch.chdir(project_dir)

    cfg = config.load_config()

    assert cfg == ScribeMdConfig(_sources=["built-in defaults"])


def test_bad_numeric_type_raises(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / config.PROJECT_CONFIG_NAME).write_text(
        "[defaults]\nchunk_seconds = \"not-a-number\"\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "USER_CONFIG_PATH", tmp_path / "missing.toml")
    monkeypatch.chdir(project_dir)

    with pytest.raises(ValueError):
        config.load_config()


def test_config_as_toml_redacts_hf_token():
    cfg = ScribeMdConfig(hf_token="hf_secret_token")

    rendered = config.config_as_toml(cfg)

    assert "hf_secret_token" not in rendered
    assert 'hf_token = "<set>"' in rendered


# Task 5: [gpu].gpus setting
import tomllib
from scribe_md.config import _apply_toml, config_as_toml


def test_gpu_section_parsed():
    cfg = ScribeMdConfig()
    _apply_toml(cfg, {"gpu": {"gpus": "auto"}}, "test")
    assert cfg.gpus == "auto"


def test_gpus_default_empty_and_round_trips_through_toml():
    cfg = ScribeMdConfig()
    assert cfg.gpus == ""
    rendered = config_as_toml(cfg)
    assert "[gpu]" in rendered
    tomllib.loads(rendered)  # must remain valid TOML
