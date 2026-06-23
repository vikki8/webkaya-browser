import unittest

from webkaya import CodeAgent, CodeGenResult, MemorySnapshotStore, Sandbox


class ScriptedProvider:
    """Returns a scripted code snippet per call; records prompts."""

    name = "scripted"

    def __init__(self, snippets):
        self.snippets = snippets
        self.calls = 0
        self.prompts = []

    def generate_code(self, system, prompt):
        self.prompts.append(prompt)
        code = self.snippets[min(self.calls, len(self.snippets) - 1)]
        self.calls += 1
        return CodeGenResult(code=code, explanation=f"attempt {self.calls}", input_tokens=5, output_tokens=3)


def make_agent(snippets, max_attempts=3):
    provider = ScriptedProvider(snippets)
    sandbox = Sandbox.create(policy={"cold_start_ms": 0, "retry_count": 0}, store=MemorySnapshotStore())
    return CodeAgent(provider, sandbox, max_attempts=max_attempts), provider, sandbox


class TestCodeAgent(unittest.TestCase):
    def test_runs_generated_code(self):
        agent, _provider, _sb = make_agent(["ctx.state['n'] = 41 + 1\nreturn ctx.state['n']"])
        outcome = agent.run("compute the answer")
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.result.value, 42)
        self.assertEqual(len(outcome.attempts), 1)
        self.assertEqual((outcome.input_tokens, outcome.output_tokens), (5, 3))

    def test_repairs_after_failure(self):
        agent, provider, _sb = make_agent(["raise RuntimeError('boom')", "return 7"])
        outcome = agent.run("do the thing")
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.result.value, 7)
        self.assertEqual(len(outcome.attempts), 2)
        self.assertEqual((outcome.input_tokens, outcome.output_tokens), (10, 6))
        # The repair prompt carried the previous error.
        self.assertIn("boom", provider.prompts[1])

    def test_gives_up_after_max_attempts(self):
        agent, _provider, _sb = make_agent(["raise RuntimeError('always')"], max_attempts=2)
        outcome = agent.run("impossible")
        self.assertFalse(outcome.ok)
        self.assertEqual(len(outcome.attempts), 2)
        self.assertIn("always", outcome.result.error)

    def test_passes_args_to_the_guest(self):
        agent, _provider, _sb = make_agent(["return ctx.args['a'] + ctx.args['b']"])
        outcome = agent.run("add them", args={"a": 40, "b": 2})
        self.assertEqual(outcome.result.value, 42)

    def test_failed_attempt_does_not_commit_state(self):
        agent, _provider, sandbox = make_agent(
            ["ctx.state['x'] = 'bad'\nraise RuntimeError('boom')", "ctx.state['x'] = 'good'\nreturn 1"]
        )
        outcome = agent.run("set x safely")
        self.assertTrue(outcome.ok)
        # The failed first attempt left no trace; only the successful run committed.
        self.assertEqual(sandbox.get_state(), {"x": "good"})


if __name__ == "__main__":
    unittest.main()
