from typing import Union

import tomli
from pathlib import Path

def load_config(file: Union[Path, str, None] = None):
    if file is None:
        file = Path("./config.toml")
    if isinstance(file, str):
        file = Path(file)
    try:
        with open(file, "rb") as f:
            return tomli.load(f)
    except Exception as e:
        print(e)
        raise RuntimeError(f"Cannot load config file: {file}")

def get_value(key: str, default=None):
    if get_value.cache is None:
        config_d = get_value.cache = load_config()
    else:
        config_d = get_value.cache
    return config_d[key] if key in config_d else default
get_value.cache = None

