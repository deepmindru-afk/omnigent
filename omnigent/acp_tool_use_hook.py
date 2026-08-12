"""ACP PreToolUse hook for gating tools via Omnigent policy evaluation.

This module is invoked as a subprocess by Devin (or any ACP agent) when a tool
use event occurs. It reads the hook payload from stdin, evaluates the tool call
against Omnigent's policy engine, and returns a block/allow decision on stdout.

The hook format is identical to Claude Code's native hooks, so agents reading
Claude Code hook configs will see `PreToolUse` events here. MCP tools (tool_name
matching `^mcp__.*`) are included in the gate since they are loaded and run by
the agent itself, not by the omnigent relay.

Failure handling: if the policy evaluation endpoint is unreachable or returns
an error, the hook fails closed by denying the tool call (consistent with the
runner-side default).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from omnigent.native_policy_hook import (
    evaluation_response_to_hook_output,
    fail_closed_hook_output,
    hook_payload_to_evaluation_request,
    policy_hook_reauth,
    post_evaluate_with_retry,
    read_relay_policy_config,
    relay_policy_evaluate_url,
)

_logger = logging.getLogger(__name__)

# Hook event type we gate at. PostToolUse and UserPromptSubmit are not handled
# here (ACP agents don't have UserPromptSubmit since they manage their own loop).
_PRE_TOOL_USE = "PreToolUse"

# Env vars for policy-hook routing (set by the ACP executor).
_AUTH_HEADERS_ENV = "_OMNIGENT_AUTH_HEADERS"
_RELAY_DIR_ENV = "_OMNIGENT_RELAY_DIR"


def main(argv: list[str] | None = None) -> int:
    """
    Evaluate an ACP ``PreToolUse`` hook against Omnigent policies.

    Reads the hook payload from stdin (JSON), converts it to an
    ``EvaluationRequest``, POSTs to the policy evaluate endpoint, and writes
    the decision on stdout.

    :param argv: Optional argv override (excluding program name). ``None`` reads
        :data:`sys.argv`.
    :returns: Process exit code (always 0 — blocking verdicts are expressed
        via JSON output, not exit codes).
    """
    raw_argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description="Evaluate ACP tool use against Omnigent policy")
    parser.add_argument(
        "--relay-dir",
        type=str,
        help="Path to relay directory containing tool relay config",
    )
    args = parser.parse_args(raw_argv)

    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError) as exc:
        # Malformed input: fail silently (no policy-governed session).
        _logger.warning("acp hook: failed to parse stdin JSON: %s", exc)
        return 0

    hook_event = payload.get("hook_event_name", "")
    if hook_event != _PRE_TOOL_USE:
        # Only handle PreToolUse; other events are not policy-gated here.
        return 0

    # Convert to evaluation request. Skips mcp__omnigent__* tools but includes
    # all mcp__* tools the agent loaded itself.
    eval_request = hook_payload_to_evaluation_request(hook_event, payload)
    if eval_request is None:
        # Not a policy-relevant event (e.g., an omnigent MCP tool).
        return 0

    # Resolve the policy endpoint. The executor writes relay config when
    # gating is enabled; if absent, the session is not governed.
    relay_dir = args.relay_dir or os.environ.get(_RELAY_DIR_ENV)
    if not relay_dir:
        return 0

    relay_config = read_relay_policy_config(relay_dir)
    if relay_config is None:
        return 0

    relay_url, _relay_token, _session_id = relay_config
    evaluate_url = relay_policy_evaluate_url(relay_url)

    # Build request headers from env (set by executor).
    headers = {"Content-Type": "application/json"}
    raw_auth = os.environ.get(_AUTH_HEADERS_ENV, "")
    if raw_auth:
        try:
            extra = json.loads(raw_auth)
            if isinstance(extra, dict):
                headers.update({str(k): str(v) for k, v in extra.items()})
        except (json.JSONDecodeError, ValueError):
            pass

    # POST to evaluate endpoint with reauth on 401/403.
    reauth_handler = policy_hook_reauth(relay_url, headers)

    def retry_reauth() -> dict[str, str] | None:
        return reauth_handler()

    response = post_evaluate_with_retry(
        evaluate_url,
        eval_request,
        headers,
        reauth_callback=retry_reauth,
    )

    if response is None:
        # Endpoint unreachable or returned non-2xx; fail closed.
        output = fail_closed_hook_output(hook_event)
        if output is not None:
            json.dump(output, sys.stdout)
        return 0

    # Convert evaluation response to hook output format (deny/allow decision).
    output = evaluation_response_to_hook_output(hook_event, response)
    if output is not None:
        json.dump(output, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
