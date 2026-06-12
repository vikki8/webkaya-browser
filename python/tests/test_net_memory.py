import unittest

from webkaya import (
    LoadBalancer,
    MemorySnapshotStore,
    MemoryTier,
    Sandbox,
    SandboxFabric,
    TieredMemory,
    deny_east_west_policy,
    hash_balancer,
)


def make_sandbox(memory=None):
    return Sandbox.create(
        policy={"cold_start_ms": 0, "retry_count": 0, "timeout_ms": 2000},
        store=MemorySnapshotStore(),
        memory=memory,
    )


class TestMemoryTier(unittest.TestCase):
    def test_get_set_delete(self):
        tier = MemoryTier()
        tier.set("k", "v")
        self.assertEqual(tier.get("k"), "v")
        self.assertTrue(tier.delete("k"))
        self.assertIsNone(tier.get("k"))

    def test_incr(self):
        tier = MemoryTier()
        self.assertEqual(tier.incr("hits"), 1)
        self.assertEqual(tier.incr("hits", 4), 5)

    def test_ttl_expiry_with_controllable_clock(self):
        now = {"t": 1000}
        tier = MemoryTier(now=lambda: now["t"])
        tier.set("session", "abc", ttl_ms=500)
        self.assertEqual(tier.get("session"), "abc")
        self.assertEqual(tier.ttl("session"), 500)
        now["t"] = 1600
        self.assertIsNone(tier.get("session"))
        self.assertEqual(tier.ttl("session"), -2)

    def test_glob_keys(self):
        tier = MemoryTier()
        tier.set("user:1", "a")
        tier.set("user:2", "b")
        tier.set("session:1", "c")
        self.assertEqual(sorted(tier.keys("user:*")), ["user:1", "user:2"])
        self.assertEqual(len(tier.keys()), 3)


class TestTieredMemory(unittest.TestCase):
    def test_local_isolated_global_shared(self):
        memory = TieredMemory()
        a = memory.binding_for("a")
        b = memory.binding_for("b")
        a.local.set("secret", "a-only")
        b.local.set("secret", "b-only")
        self.assertEqual(a.local.get("secret"), "a-only")
        self.assertEqual(b.local.get("secret"), "b-only")
        a.shared.set("shared", "visible")
        self.assertEqual(b.shared.get("shared"), "visible")


class TestFabric(unittest.TestCase):
    def test_delivers_requests(self):
        fabric = SandboxFabric()
        a = make_sandbox()
        b = make_sandbox()
        addr_a = fabric.join(a, name="a")
        addr_b = fabric.join(b, name="b",
                             handler="return {'greeting': 'hi', 'caller': ctx.args['from']}")
        response = fabric.request(addr_a, addr_b, payload={"ping": True})
        self.assertTrue(response.ok)
        self.assertEqual(response.status, 200)
        self.assertEqual(response.body, {"greeting": "hi", "caller": addr_a})
        self.assertEqual(fabric.delivered_by_dst.get(addr_b), 1)

    def test_unknown_destination(self):
        fabric = SandboxFabric()
        addr_a = fabric.join(make_sandbox())
        response = fabric.request(addr_a, 999, payload={})
        self.assertEqual(response.status, 404)

    def test_handler_failure_is_500(self):
        fabric = SandboxFabric()
        addr_a = fabric.join(make_sandbox())
        addr_b = fabric.join(make_sandbox(), handler="raise RuntimeError('handler boom')")
        response = fabric.request(addr_a, addr_b, payload={})
        self.assertEqual(response.status, 500)
        self.assertIn("handler boom", response.error)


class TestNetworkPolicy(unittest.TestCase):
    def test_deny_east_west(self):
        fabric = SandboxFabric(policy_program=deny_east_west_policy())
        addr_a = fabric.join(make_sandbox(), name="a")
        addr_b = fabric.join(make_sandbox(), name="b", handler="return 'reached b'")
        response = fabric.request(addr_a, addr_b, payload={})
        self.assertFalse(response.ok)
        self.assertEqual(response.status, 403)
        self.assertTrue(response.denied)
        self.assertEqual(fabric.dropped_by_src.get(addr_a), 1)

    def test_ingress_still_allowed(self):
        fabric = SandboxFabric(policy_program=deny_east_west_policy())
        addr_b = fabric.join(make_sandbox(), handler="return 'ok'")
        response = fabric.request(0, addr_b, payload={})
        self.assertTrue(response.ok)
        self.assertEqual(response.body, "ok")


class TestLoadBalancer(unittest.TestCase):
    def test_round_robin(self):
        fabric = SandboxFabric()
        lb = LoadBalancer(fabric)
        for i in range(2):
            lb.add_backend(fabric.join(make_sandbox(), handler=f"return {{'backend': {i}}}"))
        hits = [0, 0]
        for i in range(6):
            res = lb.handle(path="/api", payload={"n": i})
            hits[res.body["backend"]] += 1
        self.assertEqual(hits, [3, 3])

    def test_sticky_hash(self):
        fabric = SandboxFabric()
        lb = LoadBalancer(fabric, program=hash_balancer())
        for i in range(3):
            lb.add_backend(fabric.join(make_sandbox(), handler=f"return {{'backend': {i}}}"))
        first = lb.handle(path="/user/42", request_hash=42)
        again = lb.handle(path="/user/42", request_hash=42)
        self.assertEqual(again.body, first.body)

    def test_serves_static(self):
        fabric = SandboxFabric()
        lb = LoadBalancer(fabric)
        lb.serve_static("/health", {"status": "green"})
        res = lb.handle(path="/health")
        self.assertEqual(res.status, 200)
        self.assertEqual(res.body, {"status": "green"})

    def test_no_backends(self):
        res = LoadBalancer(SandboxFabric()).handle(path="/api")
        self.assertEqual(res.status, 503)

    def test_shared_global_counter_across_backends(self):
        fabric = SandboxFabric()
        memory = TieredMemory()
        lb = LoadBalancer(fabric)
        handler = "n = ctx.shared.incr('requests')\nctx.local.incr('local_hits')\nreturn {'total': n}"
        for i in range(3):
            box = Sandbox.create(policy={"cold_start_ms": 0}, store=MemorySnapshotStore(),
                                 memory=memory.binding_for(f"backend-{i}"))
            lb.add_backend(fabric.join(box, name=f"backend-{i}", handler=handler))
        totals = [lb.handle(path="/api", payload={"i": i}).body["total"] for i in range(6)]
        self.assertEqual(totals, [1, 2, 3, 4, 5, 6])
        self.assertEqual(memory.shared.get("requests"), "6")
        self.assertEqual(memory.local_for("backend-0").get("local_hits"), "2")


if __name__ == "__main__":
    unittest.main()
