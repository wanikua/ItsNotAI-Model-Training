def sanitize_label(label: str) -> str:
    """
    Convert a free-form label that means “real / human” or “fake / AI”
    into the canonical strings ``"real"`` or ``"fake"``.
    """
    _REAL  = {"real", "human", "hum"}
    _FAKE  = {"fake", "ai"}

    normalized = label.strip().lower()

    if normalized in _REAL:
        return "real"
    if normalized in _FAKE:
        return "fake"

    raise ValueError(f"Unrecognized label: {label!r}")
