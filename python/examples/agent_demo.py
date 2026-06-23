"""Run the Claude code-agent loop from Python.

The agent asks Claude to write Python, runs it in a governed sandbox over local
data, and repairs on failure — all in a few lines, no browser involved.

    pip install webkaya[claude]
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/agent_demo.py
"""

from webkaya import ClaudeProvider, CodeAgent, Sandbox


def main() -> None:
    # The data lives in the sandbox's state; the agent never receives it
    # directly — only the task text and any error reach Claude.
    sandbox = Sandbox.create(
        initial_state={
            "rows": [
                {"region": "EMEA", "revenue": 95},
                {"region": "EMEA", "revenue": 70},
                {"region": "APAC", "revenue": 110},
                {"region": "APAC", "revenue": 60},
            ]
        }
    )

    agent = CodeAgent(ClaudeProvider(), sandbox, max_attempts=3, on_log=print)
    outcome = agent.run(
        "Sum revenue per region from ctx.state['rows'] and return a dict {region: total}."
    )

    print("\nok:", outcome.ok)
    print("result:", outcome.result.value)
    print("attempts:", len(outcome.attempts))
    print("tokens:", outcome.input_tokens, "in /", outcome.output_tokens, "out")
    print("\ncode Claude wrote:\n" + outcome.code)


if __name__ == "__main__":
    main()
