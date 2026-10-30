"""Training render migration only. Recorded controls remain immutable."""
ABSOLUTE_MARKER = "<|use_abs|>"

def render_action(text):
    """Migrate only protocol marker spellings, preserving every other byte."""
    return text.replace("<use_absolute>", ABSOLUTE_MARKER).replace("<use_abs>", ABSOLUTE_MARKER)
