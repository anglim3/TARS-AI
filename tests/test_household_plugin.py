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
            "response": {"speech": {"plain": {"speech": "Turned off the kitchen lights."}}}
        })])
        said = tools.home({"text": "turn off the kitchen lights"}, http=http)
        self.assertEqual(said, "Turned off the kitchen lights.")
        method, url, kwargs = http.calls[0]
        self.assertEqual(method, "POST")
        self.assertEqual(url, "http://homeassistant.local:8123/api/conversation/process")
        self.assertEqual(kwargs["json"], {"text": "turn off the kitchen lights"})
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-token")
        self.assertEqual(kwargs["timeout"], 15)

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
