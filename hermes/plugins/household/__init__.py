"""Household tools for the Pi Hermes profile.

This is not an Amelia skill. Amelia loads only src/skills/skill_*.py.
Copy this directory to ~/.hermes/plugins/household/ and enable it there.
"""

from .tools import (
    TOOLS,
    calendar_add,
    calendar_agenda,
    home,
    tasks_add,
    tasks_complete,
    tasks_list,
)

_HANDLERS = {
    "home": home,
    "calendar_agenda": calendar_agenda,
    "calendar_create": calendar_add,
    "tasks_list": tasks_list,
    "tasks_add": tasks_add,
    "tasks_complete": tasks_complete,
}


def register(ctx):
    """Register the six household tools. Hermes calls this once at startup."""
    for schema in TOOLS:
        name = schema["name"]
        ctx.register_tool(
            name=name,
            toolset="household",
            schema=schema,
            handler=_HANDLERS[name],
        )
