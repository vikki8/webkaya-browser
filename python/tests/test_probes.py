import unittest

from webkaya import EbpfMap, MemorySnapshotStore, Sandbox
from webkaya.asm import (
    assemble,
    call,
    exit_,
    jle_imm,
    ldxdw,
    mov_imm,
    MAP_ADD,
)


def make_sandbox(**kwargs):
    return Sandbox.create(policy={"cold_start_ms": 0, "retry_count": 0}, store=MemorySnapshotStore(), **kwargs)


# map_add(fd 0, key 0, 1); return 0 (allow)
COUNTING_PROGRAM = assemble(
    [mov_imm(1, 0), mov_imm(2, 0), mov_imm(3, 1), call(MAP_ADD), mov_imm(0, 0), exit_()]
)


class TestProbeObservation(unittest.TestCase):
    def test_counts_runs_at_run_start(self):
        counters = EbpfMap()
        box = make_sandbox()
        box.attach_probe("run:start", COUNTING_PROGRAM, name="counter", maps=[counters])
        box.run("return 1")
        box.run("return 2")
        box.run("return 3")
        self.assertEqual(counters.get(0), 3)

    def test_buckets_outcomes_at_run_end(self):
        # key = ctx.ok (field 1, offset 8); count per outcome.
        program = assemble(
            [ldxdw(2, 1, 8), mov_imm(1, 0), mov_imm(3, 1), call(MAP_ADD), mov_imm(0, 0), exit_()]
        )
        outcomes = EbpfMap()
        box = make_sandbox()
        box.attach_probe("run:end", program, name="outcomes", maps=[outcomes])
        box.run("return 1")
        box.run("raise RuntimeError('boom')")
        self.assertEqual(outcomes.get(1), 1)  # one ok
        self.assertEqual(outcomes.get(0), 1)  # one failure

    def test_observes_log_and_snapshot(self):
        logs = EbpfMap()
        snaps = EbpfMap()
        box = make_sandbox()
        box.attach_probe("log", COUNTING_PROGRAM, maps=[logs])
        box.attach_probe("snapshot", COUNTING_PROGRAM, maps=[snaps])
        box.run("ctx.log('a')\nctx.log('b')\nreturn 1")
        box.snapshot("cp")
        self.assertEqual(logs.get(0), 2)
        self.assertEqual(snaps.get(0), 1)


class TestProbeAdmissionControl(unittest.TestCase):
    # Deny when code_length (field 1, offset 8) > 16.
    MAX_LEN = assemble(
        [ldxdw(2, 1, 8), mov_imm(0, 0), jle_imm(2, 16, 1), mov_imm(0, 1), exit_()]
    )

    def test_vetoes_runs_over_the_limit(self):
        box = make_sandbox(initial_state={})
        box.attach_probe("run:start", self.MAX_LEN, name="max-len")

        allowed = box.run("return 1")
        self.assertTrue(allowed.ok)

        denied = box.run("x = 1  # padding to exceed the length limit\nreturn 2")
        self.assertFalse(denied.ok)
        self.assertIn('denied by probe "max-len"', denied.error)

    def test_denied_runs_do_not_mutate_state(self):
        box = make_sandbox(initial_state={"v": "original"})
        box.attach_probe("run:start", self.MAX_LEN, name="max-len")
        box.run("ctx.state['v'] = 'changed'  # long enough to be denied by the probe")
        self.assertEqual(box.get_state(), {"v": "original"})

    def test_fail_open_by_default_fail_closed_when_set(self):
        # Out-of-bounds read always errors at runtime.
        crashing = assemble([ldxdw(0, 1, 4096), exit_()])

        lenient = make_sandbox()
        lenient.attach_probe("run:start", crashing, name="crashy")
        self.assertTrue(lenient.run("return 1").ok)

        strict = make_sandbox()
        strict.attach_probe("run:start", crashing, name="crashy", fail_closed=True)
        denied = strict.run("return 1")
        self.assertFalse(denied.ok)
        self.assertIn('denied by probe "crashy"', denied.error)

    def test_detach_stops_firing(self):
        counters = EbpfMap()
        box = make_sandbox()
        pid = box.attach_probe("run:start", COUNTING_PROGRAM, maps=[counters])
        box.run("return 1")
        self.assertTrue(box.detach_probe(pid))
        box.run("return 2")
        self.assertEqual(counters.get(0), 1)

    def test_rejects_unknown_tracepoint(self):
        box = make_sandbox()
        with self.assertRaises(ValueError):
            box.attach_probe("not:a:tracepoint", COUNTING_PROGRAM)


if __name__ == "__main__":
    unittest.main()
