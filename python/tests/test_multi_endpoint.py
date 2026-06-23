import unittest

from webkaya import (
    LoadBalancer,
    MemorySnapshotStore,
    Sandbox,
    SandboxFabric,
    TieredMemory,
    deny_east_west_policy,
)

WRITER = "ctx.shared.incr('events'); ctx.local.incr('handled'); return {'ok': True}"
READER = "ctx.local.incr('handled'); return {'events': int(ctx.shared.get('events') or '0')}"


class TestMultiEndpointFleet(unittest.TestCase):
    """Different handlers in different pools, all coordinating through one
    shared global memory — the multi-endpoint topology from the demo."""

    def setUp(self):
        self.memory = TieredMemory()
        self.fabric = SandboxFabric(policy_program=deny_east_west_policy())

        def pool(handler, size, prefix):
            lb = LoadBalancer(self.fabric)
            for i in range(size):
                box = Sandbox.create(
                    policy={"cold_start_ms": 0},
                    store=MemorySnapshotStore(),
                    memory=self.memory.binding_for(f"{prefix}-{i}"),
                )
                lb.add_backend(self.fabric.join(box, handler=handler, name=f"{prefix}-{i}"))
            return lb

        self.write_lb = pool(WRITER, 3, "writer")
        self.read_lb = pool(READER, 1, "reader")

    def test_reader_pool_sees_writes_from_writer_pool(self):
        for i in range(5):
            res = self.write_lb.handle(path="/write", payload={"n": i})
            self.assertTrue(res.ok)
        # The read pool — a different handler on a different sandbox — observes
        # the global state the write pool produced.
        audit = self.read_lb.handle(path="/read")
        self.assertEqual(audit.body, {"events": 5})
        self.assertEqual(self.memory.shared.get("events"), "5")

    def test_writes_are_distributed_across_the_writer_pool(self):
        for i in range(6):
            self.write_lb.handle(path="/write", payload={"n": i})
        handled = [int(self.memory.local_for(f"writer-{i}").get("handled")) for i in range(3)]
        self.assertEqual(sum(handled), 6)
        self.assertEqual(handled, [2, 2, 2])   # round-robin

    def test_pools_remain_isolated_east_west(self):
        writer_addr = self.fabric.addresses()[0]
        reader_addr = self.fabric.addresses()[-1]
        denied = self.fabric.request(writer_addr, reader_addr, payload={})
        self.assertEqual(denied.status, 403)
        self.assertTrue(denied.denied)


if __name__ == "__main__":
    unittest.main()
