"""
Serialization mixin for dataclasses.

Provides consistent to_dict/from_dict methods with support for:
- datetime serialization
- nested dataclasses
- optional fields
- type coercion
"""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from typing import Any, TypeVar, get_type_hints

T = TypeVar("T")


def _serialize_value(value: Any) -> Any:
    """Serialize a single value for JSON."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value, dict_factory=_serialize_dict_factory)
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    return value


def _serialize_dict_factory(items: list[tuple[str, Any]]) -> dict[str, Any]:
    """Dict factory for asdict that handles datetime and nested types."""
    return {k: _serialize_value(v) for k, v in items}


def _deserialize_value(value: Any, type_hint: Any) -> Any:
    """Deserialize a value based on type hint."""
    if value is None:
        return None

    # Handle datetime
    if type_hint is datetime or (hasattr(type_hint, "__origin__") and type_hint.__origin__ is datetime):
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return value

    # Handle Optional types
    origin = getattr(type_hint, "__origin__", None)
    if origin is type(None):
        return None

    # Handle Union (including Optional)
    if hasattr(type_hint, "__args__"):
        # Try datetime first for Union[datetime, None]
        for arg in type_hint.__args__:
            if arg is datetime and isinstance(value, str):
                try:
                    return datetime.fromisoformat(value)
                except ValueError:
                    pass

    return value


class SerializableMixin:
    """
    Mixin that provides to_dict and from_dict for dataclasses.

    Usage:
        @dataclass
        class MyData(SerializableMixin):
            name: str
            timestamp: datetime
            count: int = 0

        data = MyData(name="test", timestamp=datetime.now())
        d = data.to_dict()  # {"name": "test", "timestamp": "2024-01-01T...", "count": 0}
        restored = MyData.from_dict(d)
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, handling datetime and nested dataclasses."""
        return asdict(self, dict_factory=_serialize_dict_factory)  # type: ignore

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Create instance from dictionary, handling datetime parsing."""
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass")

        # Get type hints for the class
        try:
            hints = get_type_hints(cls)
        except Exception:
            hints = {}

        # Build kwargs, only including fields that exist
        kwargs = {}
        for field_info in fields(cls):
            name = field_info.name
            if name not in data:
                continue

            value = data[name]
            type_hint = hints.get(name)

            if type_hint:
                value = _deserialize_value(value, type_hint)

            kwargs[name] = value

        return cls(**kwargs)  # type: ignore


def to_dict(obj: Any) -> dict[str, Any]:
    """
    Standalone function to serialize a dataclass to dict.

    Useful when you can't modify the class to add the mixin.
    """
    if not is_dataclass(obj) or isinstance(obj, type):
        raise TypeError(f"{type(obj).__name__} is not a dataclass instance")
    return asdict(obj, dict_factory=_serialize_dict_factory)
