"""Stands in for anthropic.Anthropic() and sends the request through the local Claude Code login instead of an API key."""
import asyncio
from types import SimpleNamespace

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query


class _Messages:
    def parse(self, model, max_tokens, system, messages, output_format):
        return asyncio.run(self._parse(model, system, messages[-1]["content"], output_format))

    async def _parse(self, model, system, prompt, output_format):
        options = ClaudeAgentOptions(
            model=model,
            system_prompt=system,
            tools=[],
            setting_sources=[],
            max_turns=3,
            output_format={"type": "json_schema", "schema": output_format.model_json_schema()},
        )
        result = None
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage):
                result = message
        if result is None or result.structured_output is None:
            raise RuntimeError(f"no structured answer: {getattr(result, 'errors', None) or getattr(result, 'result', None)}")
        usage = result.usage or {}
        return SimpleNamespace(
            parsed_output=output_format.model_validate(result.structured_output),
            usage=SimpleNamespace(input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0)),
        )


class LoginClient:
    def __init__(self):
        self.messages = _Messages()
