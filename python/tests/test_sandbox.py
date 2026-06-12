import unittest

from webkaya import MemorySnapshotStore, Sandbox, TieredMemory


def make_sandbox(initial_state=None, memory=None):
    return Sandbox.create(
        policy={"cold_start_ms": 0, "retry_count": 0, "timeout_ms": 2000},
        initial_state=initial_state or {},
        store=MemorySnapshotStore(),
        memory=memory,
    )


class TestSandboxRun(unittest.TestCase):
    def test_runs_guest_against_state(self):
        box = make_sandbox({"counter": 1})
        result = box.run("ctx.state['counter'] += 41\nreturn ctx.state['counter']")
        self.assertTrue(result.ok)
        self.assertEqual(result.value, 42)
        self.assertEqual(box.get_state(), {"counter": 42})

    def test_args_and_logs(self):
        box = make_sandbox()
        result = box.run("ctx.log('hello ' + ctx.args['who'])\nreturn ctx.args['who']",
                         args={"who": "agent"})
        self.assertEqual(result.value, "agent")
        self.assertEqual(result.logs, ["hello agent"])

    def test_disallowed_code_does_not_mutate_state(self):
        box = make_sandbox({"safe": True})
        result = box.run("import os\nreturn 1")
        self.assertFalse(result.ok)
        self.assertIn("disallowed token", result.error)
        self.assertEqual(box.get_state(), {"safe": True})

    def test_failed_run_does_not_commit_state(self):
        box = make_sandbox({"value": "original"})
        result = box.run("ctx.state['value'] = 'corrupted'\nraise RuntimeError('boom')")
        self.assertFalse(result.ok)
        self.assertIn("boom", result.error)
        self.assertEqual(box.get_state(), {"value": "original"})

    def test_memory_budget(self):
        box = make_sandbox()
        result = box.run("return 1", estimated_memory_mb=100000)
        self.assertFalse(result.ok)
        self.assertIn("above sandbox budget", result.error)


class TestSnapshotsAndForks(unittest.TestCase):
    def test_snapshot_and_restore(self):
        store = MemorySnapshotStore()
        box = Sandbox.create(policy={"cold_start_ms": 0}, initial_state={"n": 0}, store=store)
        box.run("ctx.state['n'] = 7")
        snap = box.snapshot("after-seven")

        restored = Sandbox.restore(snap.id, store=store)
        self.assertEqual(restored.get_state(), {"n": 7})
        self.assertEqual(restored.parent_snapshot_id, snap.id)

    def test_forks_diverge(self):
        box = make_sandbox({"branch": "parent"})
        fork = box.fork()
        fork.run("ctx.state['branch'] = 'fork'")
        self.assertEqual(fork.get_state(), {"branch": "fork"})
        self.assertEqual(box.get_state(), {"branch": "parent"})

    def test_auto_snapshot_cadence(self):
        store = MemorySnapshotStore()
        box = Sandbox.create(
            policy={"cold_start_ms": 0, "snapshot_every_n_runs": 2},
            initial_state={},
            store=store,
        )
        box.run("ctx.state['a'] = 1")
        self.assertEqual(len(store.list()), 0)
        box.run("ctx.state['b'] = 2")
        self.assertEqual(len(store.list()), 1)


class TestReplay(unittest.TestCase):
    def test_reproduces_run_sequence(self):
        box = make_sandbox({"total": 0})
        box.run("ctx.state['total'] += ctx.args", args=10)
        box.run("ctx.state['total'] *= 2")

        results, final_state = box.replay()
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.ok for r in results))
        self.assertEqual(final_state, {"total": 20})
        self.assertEqual(final_state, box.get_state())


class TestDispose(unittest.TestCase):
    def test_rejects_use_after_dispose(self):
        box = make_sandbox()
        box.dispose()
        with self.assertRaisesRegex(RuntimeError, "disposed"):
            box.run("return 1")


class TestTieredMemoryAccess(unittest.TestCase):
    def test_guest_uses_local_and_shared_tiers(self):
        memory = TieredMemory()
        box = make_sandbox(memory=memory.binding_for("box-1"))
        box.run("ctx.shared.incr('hits')\nctx.local.set('name', 'agent')\nreturn ctx.shared.get('hits')")
        self.assertEqual(memory.shared.get("hits"), "1")
        self.assertEqual(memory.local_for("box-1").get("name"), "agent")


if __name__ == "__main__":
    unittest.main()
