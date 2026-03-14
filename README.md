# amplifier-module-hooks-dynamic-context

Dynamic context loading hook for Amplifier. Loads capability context on demand
rather than statically at session start.

## How it works

A **classifier** inspects each user message for trigger keywords and activates
matching context blocks. The **loader hook** then injects those blocks
ephemerally before each LLM call — keeping the system prompt lean until topics
are actually needed.

Two components ship as entry points in the same package:

| Entry point | Type | Purpose |
|---|---|---|
| `hooks-dynamic-context` | hook | Auto-classifies + injects |
| `tool-load-context` | tool | Manual `load_context` tool for the LLM |

## Manifest format

```yaml
context_blocks:
  - name: imessage-knowledge
    description: "iMessage database queries, AppleScript send/receive, contact resolution"
    path: "/absolute/path/to/imessage-knowledge.md"
    triggers: ["message", "text", "imessage", "sms", "texts", "texted"]

  - name: calendar-knowledge
    description: "Google Calendar API, event creation, recurring events, attendees"
    path: "/absolute/path/to/calendar-knowledge.md"
    triggers: ["calendar", "event", "schedule", "meeting", "appointment"]
```

## Bundle configuration

```yaml
modules:
  - source: path/to/amplifier-module-hooks-dynamic-context
    name: hooks-dynamic-context
    config:
      manifest_path: ".amplifier/context/manifest.yaml"  # relative to working_dir
      priority: 20  # optional, default 20

  - source: path/to/amplifier-module-hooks-dynamic-context
    name: tool-load-context
```

## Classifier behaviour

- **Trigger matching**: single-word triggers use word-boundary regex (`\b`);
  multi-word triggers use substring match.
- **Accumulative**: once a block is activated it stays active for the session.
- **Fallback**: the LLM can call `load_context(name="...")` explicitly if the
  classifier missed something, or `load_context(name="list")` to see all blocks.

## Graceful degradation

If `manifest_path` is missing, the file doesn't exist, or the manifest has no
valid blocks, the hook logs a warning and does nothing. All other modules
continue to work normally.

## Development

```bash
uv sync
uv run pytest tests/ -v
```
