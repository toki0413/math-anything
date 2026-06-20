"""Config command implementation."""

from ..config import get_config


def cmd_config(args):
    """Manage configuration."""
    cfg = get_config()

    if args.action == "list":
        print("Current configuration:")
        print(f"  Config file: {cfg.config_file}")
        print("")
        _print_dict(cfg.data)
        return 0

    elif args.action == "path":
        print(f"Config file: {cfg.config_file}")
        return 0

    elif args.action == "get":
        if not args.key:
            print("Error: key required for 'get'")
            return 1
        value = cfg.get(args.key)
        if value is not None:
            print(value)
        else:
            print(f"Key '{args.key}' not found")
        return 0

    elif args.action == "set":
        if not args.key or args.value is None:
            print("Error: key and value required for 'set'")
            return 1
        # Try to parse as number/bool
        parsed = _parse_value(args.value)
        cfg.set(args.key, parsed)
        print(f"Set {args.key} = {parsed}")
        return 0

    return 0


def _print_dict(d, indent=0):
    for k, v in d.items():
        if k == "api_key" and v:
            v = "***" + v[-4:] if len(v) > 4 else "****"
        if isinstance(v, dict):
            print("  " * indent + f"{k}:")
            _print_dict(v, indent + 1)
        else:
            print("  " * indent + f"{k}: {v}")


def _parse_value(value: str):
    """Parse string value to appropriate type."""
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "null" or lower == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
