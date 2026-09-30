from types import SimpleNamespace

import yaml


def load_config(config_path):
    """Load a YAML config file so values can be read as attributes.

    For example, config.wine_map.width instead of config["wine_map"]["width"].
    """
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return _to_namespace(config)


def _to_namespace(value):
    """Turn nested dictionaries into nested SimpleNamespace objects."""
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: _to_namespace(item) for key, item in value.items()}
        )
    return value
