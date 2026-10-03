"""Domain-specific exceptions so callers can distinguish user errors from bugs."""


class AutoforgeError(Exception):
    """Base class for all expected, user-facing errors."""


class DataLoadError(AutoforgeError):
    """The dataset could not be read or is unusable."""


class ConfigError(AutoforgeError):
    """The requested experiment configuration is invalid."""


class SchemaError(AutoforgeError):
    """New data does not match the schema the model was trained on."""
