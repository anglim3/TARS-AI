"""Household plugin checks. No network and no tokens."""

import importlib
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN_PARENT = os.path.join(ROOT, "hermes", "plugins")
if PLUGIN_PARENT not in sys.path:
    sys.path.insert(0, PLUGIN_PARENT)

import household
import household.tools as tools


class Response:
    def __init__(self, payload, ok=True):
        self._payload = payload
        self.ok = ok

    def json(self):
        return self._payload


class RecordingHttp:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def _take(self, method, url, kwargs):
        self.calls.append((method, url, kwargs))
        if not self.responses:
            raise AssertionError(f"unexpected {method} {url}")
        return self.responses.pop(0)

    def get(self, url, **kwargs):
        return self._take("GET", url, kwargs)

    def post(self, url, **kwargs):
        return self._take("POST", url, kwargs)


class Ctx:
    def __init__(self):
        self.tools = []

    def register_tool(self, name, toolset, schema, handler):
        self.tools.append((name, toolset, schema, handler))


class HouseholdTests(unittest.TestCase):
    def setUp(self):
        self._saved = dict(os.environ)
        for name in (
            "HA_TOKEN",
            "HA_URL",
            "GOOGLE_OAUTH_CLIENT_ID",
            "GOOGLE_OAUTH_CLIENT_SECRET",
            "GOOGLE_OAUTH_REFRESH_TOKEN",
            "TODOIST_API_TOKEN",
            "CALENDAR_ID",
            "CALENDAR_TIMEZONE",
        ):
            os.environ.pop(name, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved)

    def test_register_lists_the_six_tools(self):
        ctx = Ctx()
        household.register(ctx)
        names = [item[0] for item in ctx.tools]
        self.assertEqual(names, [
            "home",
            "calendar_agenda",
            "calendar_create",
            "tasks_list",
            "tasks_add",
            "tasks_complete",
        ])
        self.assertTrue(all(item[1] == "household" for item in ctx.tools))

    def test_missing_env_does_not_call_http(self):
        http = RecordingHttp([])
        self.assertEqual(tools.home({"text": "turn off the kitchen lights"}, http=http), "Home Assistant is not configured.")
        self.assertEqual(tools.calendar_agenda({"when": "today"}, http=http), "Google Calendar is not configured.")
        self.assertEqual(tools.tasks_list({"filter": "today"}, http=http), "Todoist is not configured.")
        self.assertEqual(http.calls, [])

    def test_home_posts_the_phrase_unchanged(self):
        os.environ["HA_TOKEN"] = "test-token"
        os.environ["HA_URL"] = "http://homeassistant.local:8123"
        http = RecordingHttp([Response({
            "response": {"speech": {"plain": {"speech": "The kitchen is warm."}}}
        })])
        said = tools.home({"text": "how warm is the kitchen"}, http=http)
        self.assertEqual(said, "The kitchen is warm.")
        method, url, kwargs = http.calls[0]
        self.assertEqual(method, "POST")
        self.assertEqual(url, "http://homeassistant.local:8123/api/conversation/process")
        self.assertEqual(kwargs["json"], {"text": "how warm is the kitchen"})
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-token")
        self.assertEqual(kwargs["timeout"], 15)

    def _states(self):
        return [
            {
                "entity_id": "light.ceiling",
                "state": "off",
                "attributes": {"friendly_name": "Ceiling"},
            },
            {
                "entity_id": "light.porch",
                "state": "on",
                "attributes": {"friendly_name": "Porch"},
            },
            {
                "entity_id": "switch.fan",
                "state": "off",
                "attributes": {"friendly_name": "Fan"},
            },
        ]

    def test_lights_on_controls_every_light_from_the_entity_list(self):
        os.environ["HA_TOKEN"] = "test-token"
        os.environ["HA_URL"] = "http://homeassistant.local:8123"
        # A non-error conversation reply must not skip the entity-list call.
        http = RecordingHttp([
            Response(self._states()),
            Response([]),
            Response({
                "response": {
                    "response_type": "action_done",
                    "speech": {"plain": {"speech": "Turned on the light"}},
                }
            }),
        ])
        said = tools.home({"text": "turn the lights on"}, http=http)
        self.assertEqual(said, "Turned the lights on.")
        self.assertEqual([call[0] for call in http.calls], ["GET", "POST"])
        method, url, kwargs = http.calls[1]
        self.assertEqual(url, "http://homeassistant.local:8123/api/services/light/turn_on")
        self.assertEqual(set(kwargs["json"]["entity_id"]), {"light.ceiling", "light.porch"})
        self.assertNotIn("switch.fan", kwargs["json"]["entity_id"])

    def test_named_light_is_controlled_alone(self):
        os.environ["HA_TOKEN"] = "test-token"
        os.environ["HA_URL"] = "http://homeassistant.local:8123"
        http = RecordingHttp([Response(self._states()), Response([])])
        said = tools.home({"text": "turn the porch off"}, http=http)
        self.assertEqual(said, "Turned Porch off.")
        self.assertEqual(http.calls[1][2]["json"]["entity_id"], ["light.porch"])
        self.assertIn("/api/services/light/turn_off", http.calls[1][1])

    def test_named_switch_is_controlled_alone(self):
        os.environ["HA_TOKEN"] = "test-token"
        os.environ["HA_URL"] = "http://homeassistant.local:8123"
        http = RecordingHttp([Response(self._states()), Response([])])
        said = tools.home({"text": "turn the fan on"}, http=http)
        self.assertEqual(said, "Turned Fan on.")
        self.assertEqual(http.calls[1][2]["json"]["entity_id"], ["switch.fan"])
        self.assertIn("/api/services/switch/turn_on", http.calls[1][1])

    def test_roster_uses_live_names(self):
        os.environ["HA_TOKEN"] = "test-token"
        os.environ["HA_URL"] = "http://homeassistant.local:8123"
        http = RecordingHttp([Response(self._states())])
        text = tools.device_roster(http=http)
        self.assertIn("Ceiling (light, off)", text)
        self.assertIn("Porch (light, on)", text)
        self.assertIn("Fan (switch, off)", text)
        self.assertIn("every light", text)
        self.assertNotIn("Dimmer", text)
        self.assertNotIn("Outlet", text)

    def test_tasks_complete_refuses_two_matches(self):
        os.environ["TODOIST_API_TOKEN"] = "test-token"
        http = RecordingHttp([Response([
            {"id": "1", "content": "take out trash"},
            {"id": "2", "content": "take out recycling"},
        ])])
        said = tools.tasks_complete({"title": "take out"}, http=http)
        self.assertIn("More than one match", said)
        self.assertEqual(len(http.calls), 1)
        self.assertEqual(http.calls[0][0], "GET")

    def test_not_an_amelia_skill_path(self):
        skill_dir = os.path.join(ROOT, "src", "skills")
        names = os.listdir(skill_dir)
        self.assertFalse(any(name.startswith("skill_household") for name in names))
        self.assertFalse(any(name.startswith("skill_homebridge") for name in names))


if __name__ == "__main__":
    unittest.main()
