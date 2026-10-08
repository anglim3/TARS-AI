"""Household tools for the Pi Hermes profile.

This is not an Amelia skill. Amelia loads only src/skills/skill_*.py.
Copy this directory to ~/.hermes/plugins/household/ and enable it there.
"""

from .tools import (
    TOOLS,
    calendar_add,
    calendar_agenda,
    home,
    ha_call_service,
    ha_services,
    ha_states,
    tasks_add,
    tasks_complete,
    tasks_list,
    weather,
)

_HANDLERS = {
    "home": home,
    "calendar_agenda": calendar_agenda,
    "calendar_create": calendar_add,
    "tasks_list": tasks_list,
    "tasks_add": tasks_add,
    "tasks_complete": tasks_complete,
    "ha_states": ha_states,
    "ha_call_service": ha_call_service,
    "ha_services": ha_services,
    "weather": weather,
}


def register(ctx):
    """Register the household tools. Hermes calls this once at startup."""
    for schema in TOOLS:
        name = schema["name"]
        ctx.register_tool(
            name=name,
            toolset="household",
            schema=schema,
            handler=_HANDLERS[name],
        )
