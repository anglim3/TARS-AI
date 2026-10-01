Household plugin

Hermes tools for the house, the calendar, and the task list. Amelia does not
load this directory. SkillManager only imports src/skills/skill_*.py, and this
tree is not under that path. Do not add a skill_*.py for these tools, and do
not turn on [SKILL:home_assistant] for the voice path.

Copy this folder to ~/.hermes/plugins/household/ on the Pi and enable
household in that Hermes profile. Confirm a POST /v1/responses turn lists
home, calendar_agenda, calendar_create, tasks_list, tasks_add, and
tasks_complete.

Secrets stay in ~/.hermes/.env. Non-secret settings (HA_URL, CALENDAR_ID,
CALENDAR_TIMEZONE) can live there too. Nothing in this folder is a token.

What only Jack can do

1. In the xAI console, copy the TARS voice id into Amelia's .env as
   XAI_TTS_VOICE_ID. Put the xAI speech key in XAI_API_KEY.
2. On the Pi Hermes profile, set API_SERVER_ENABLED, bind API_SERVER_HOST to
   loopback, and set API_SERVER_KEY. Put that same secret in Amelia's .env
   as HERMES_API_KEY.
3. Home Assistant: create a long-lived access token. Set HA_TOKEN and HA_URL.
   If Homebridge accessories should answer to voice, pair them into Home
   Assistant with the HomeKit Device integration. Spoken names must match
   Home Assistant friendly names. Skip a Homebridge controller unless Home
   Assistant is not in the house.
4. Todoist: create a personal API token on the integrations page. Set
   TODOIST_API_TOKEN. OAuth is not required for this one box.
5. Google Calendar: in Google Cloud, create a desktop OAuth client, enable
   the Calendar API, and consent once with scope
   https://www.googleapis.com/auth/calendar.events. Store
   GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET, and
   GOOGLE_OAUTH_REFRESH_TOKEN on the Pi. Set CALENDAR_ID (default primary)
   and CALENDAR_TIMEZONE (an IANA name). A service account cannot see a
   personal Gmail calendar.
6. Restart the Pi Hermes gateway after the env file and the plugin are in
   place. Leave the Mac Mini off this path.

Do not commit .env, config.ini, or any of those values.
