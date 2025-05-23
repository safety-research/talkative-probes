"""Flexible epoch/step scheduling parser with suffix notation."""

import re
from typing import Union, Dict, Any
from dataclasses import dataclass

@dataclass
class ScheduleSpec:
    """Represents a parsed schedule specification."""
    value: int
    unit: str  # 'steps' or 'epochs'
    
    def __str__(self):
        return f"{self.value}{self.unit[0]}"  # e.g., "1000s" or "5e"


def parse_schedule_value(value: Union[str, int, None]) -> ScheduleSpec:
    """Parse a schedule value with optional suffix.
    
    Args:
        value: Can be:
            - int: Interpreted as steps (legacy behavior)
            - str with suffix: "1000s" (steps), "5e" (epochs), "2000steps", "10epochs"
            - None: Returns None
    
    Returns:
        ScheduleSpec with parsed value and unit
        
    Examples:
        parse_schedule_value("1000s") -> ScheduleSpec(1000, "steps")
        parse_schedule_value("5e") -> ScheduleSpec(5, "epochs")
        parse_schedule_value("2000steps") -> ScheduleSpec(2000, "steps")
        parse_schedule_value("10epochs") -> ScheduleSpec(10, "epochs")
        parse_schedule_value(1000) -> ScheduleSpec(1000, "steps")  # legacy
    """
    if value is None:
        return None
    
    if isinstance(value, int):
        # Legacy: bare integers are interpreted as steps
        return ScheduleSpec(value, "steps")
    
    if isinstance(value, str):
        # Try to match patterns like "1000s", "5e", "2000steps", "10epochs"
        patterns = [
            (r'^(\d+)s$', 'steps'),           # "1000s"
            (r'^(\d+)e$', 'epochs'),          # "5e"
            (r'^(\d+)steps?$', 'steps'),      # "1000step" or "1000steps"
            (r'^(\d+)epochs?$', 'epochs'),    # "5epoch" or "5epochs"
        ]
        
        for pattern, unit in patterns:
            match = re.match(pattern, value.lower())
            if match:
                return ScheduleSpec(int(match.group(1)), unit)
        
        # Try parsing as plain integer (legacy string format)
        try:
            return ScheduleSpec(int(value), "steps")
        except ValueError:
            pass
    
    raise ValueError(f"Invalid schedule value format: {value}. "
                     f"Expected formats: '1000s', '5e', '2000steps', '10epochs', or integer.")


def parse_schedule_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a config dict and convert suffixed schedule values.
    
    This function walks through the config and converts any values that look like
    schedule specifications into detailed format.
    
    Args:
        config: Configuration dictionary that may contain suffixed values
        
    Returns:
        New configuration dictionary with parsed schedule specifications
    """
    def parse_recursive(obj):
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key.endswith('_at_step') or key.endswith('_at_epoch') or key.endswith('_steps') or key.endswith('_epochs'):
                    # Skip keys that are already explicitly step/epoch specific
                    result[key] = value
                elif key in ['unfreeze_at', 'freeze_at', 'warmup_steps', 'warmup_epochs', 'interval']:
                    # Parse scheduling keys
                    try:
                        parsed = parse_schedule_value(value)
                        if parsed:
                            # Store both the parsed spec and the original value for compatibility
                            result[key] = value  # Keep original
                            result[f"{key}_parsed"] = {
                                'value': parsed.value,
                                'unit': parsed.unit
                            }
                        else:
                            result[key] = value
                    except ValueError as e:
                        print(f"Warning: Failed to parse schedule value for {key}: {e}")
                        result[key] = value
                else:
                    result[key] = parse_recursive(value)
            return result
        elif isinstance(obj, list):
            return [parse_recursive(item) for item in obj]
        else:
            return obj
    
    return parse_recursive(config)


def resolve_schedule_at_step(spec: ScheduleSpec, current_step: int, current_epoch: int) -> bool:
    """Check if a schedule spec should trigger at the current step/epoch.
    
    Args:
        spec: Parsed schedule specification
        current_step: Current training step
        current_epoch: Current training epoch
        
    Returns:
        True if the schedule should trigger now
    """
    if spec is None:
        return False
    
    if spec.unit == "steps":
        return current_step >= spec.value
    elif spec.unit == "epochs":
        return current_epoch >= spec.value
    else:
        raise ValueError(f"Unknown schedule unit: {spec.unit}")


def get_schedule_value_for_logging(spec: ScheduleSpec) -> str:
    """Get a human-readable string for logging purposes."""
    if spec is None:
        return "never"
    return f"{spec.value} {spec.unit}"


# Example usage and test cases
def test_parser():
    """Test the schedule parser with various inputs."""
    test_cases = [
        ("1000s", ScheduleSpec(1000, "steps")),
        ("5e", ScheduleSpec(5, "epochs")),
        ("2000steps", ScheduleSpec(2000, "steps")),
        ("10epochs", ScheduleSpec(10, "epochs")),
        ("1step", ScheduleSpec(1, "steps")),
        ("1epoch", ScheduleSpec(1, "epochs")),
        (1000, ScheduleSpec(1000, "steps")),
        ("1000", ScheduleSpec(1000, "steps")),
    ]
    
    print("Testing schedule parser:")
    for input_val, expected in test_cases:
        try:
            result = parse_schedule_value(input_val)
            success = (result.value == expected.value and result.unit == expected.unit)
            print(f"  {input_val} -> {result} {'✓' if success else '✗'}")
        except Exception as e:
            print(f"  {input_val} -> ERROR: {e}")
    
    # Test config parsing
    test_config = {
        "freeze_schedule": {
            "unfreeze_at": "1000s",
            "warmup_steps": "100s",
            "other_param": 42,
            "nested": {
                "interval": "5e"
            }
        }
    }
    
    parsed_config = parse_schedule_config(test_config)
    print(f"\nTest config parsing:")
    print(f"Original: {test_config}")
    print(f"Parsed: {parsed_config}")


if __name__ == "__main__":
    test_parser()