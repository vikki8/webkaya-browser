import time
import unittest

from webkaya import KVStore, MemoryTier, RedisMemoryTier, Sandbox, TieredMemory


class FakeRedis:
    """Minimal in-memory stand-in for the redis-py subset RedisMemoryTier uses
    (decode_responses=True semantics). Lets us test the adapter without a
    server; a real Redis is a drop-in via RedisMemoryTier(url=...)."""

    def __init__(self):
        self._d = {}            # key -> str value
        self._exp = {}          # key -> expiry epoch ms

    def _live(self, key):
        exp = self._exp.get(key)
        if exp is not None and exp <= time.time() * 1000:
            self._d.pop(key, None)
            self._exp.pop(key, None)
            return False
        return key in self._d

    def get(self, key):
        return self._d[key] if self._live(key) else None

    def set(self, key, value, px=None):
        self._d[key] = str(value)
        if px is not None:
            self._exp[key] = time.time() * 1000 + px
        else:
            self._exp.pop(key, None)

    def delete(self, *keys):
        n = 0
        for key in keys:
            if self._d.pop(key, None) is not None:
                n += 1
            self._exp.pop(key, None)
        return n

    def incrby(self, key, amount):
        base = int(self._d[key]) if self._live(key) else 0
        self._d[key] = str(base + amount)
        return base + amount

    def pexpire(self, key, ttl_ms):
        if not self._live(key):
            return 0
        self._exp[key] = time.time() * 1000 + ttl_ms
        return 1

    def pttl(self, key):
        if not self._live(key):
            return -2
        exp = self._exp.get(key)
        return -1 if exp is None else max(0, int(exp - time.time() * 1000))

    def scan_iter(self, match=None):
        import fnmatch
        for key in list(self._d.keys()):
            if self._live(key) and (match is None or fnmatch.fnmatch(key, match)):
                yield key


def tier():
    return RedisMemoryTier(client=FakeRedis(), namespace="t")


class TestRedisMemoryTier(unittest.TestCase):
    def test_satisfies_the_kvstore_protocol(self):
        self.assertIsInstance(tier(), KVStore)
        self.assertIsInstance(MemoryTier(), KVStore)

    def test_get_set_delete(self):
        t = tier()
        t.set("k", "v")
        self.assertEqual(t.get("k"), "v")
        self.assertTrue(t.delete("k"))
        self.assertIsNone(t.get("k"))

    def test_incr_is_atomic_counter(self):
        t = tier()
        self.assertEqual(t.incr("hits"), 1)
        self.assertEqual(t.incr("hits", 4), 5)
        self.assertEqual(t.incr("hits", -2), 3)
        self.assertEqual(t.get("hits"), "3")

    def test_namespacing_isolates_stores(self):
        client = FakeRedis()
        a = RedisMemoryTier(client=client, namespace="a")
        b = RedisMemoryTier(client=client, namespace="b")
        a.set("k", "from-a")
        b.set("k", "from-b")
        self.assertEqual(a.get("k"), "from-a")
        self.assertEqual(b.get("k"), "from-b")
        a.flush()
        self.assertIsNone(a.get("k"))
        self.assertEqual(b.get("k"), "from-b")  # flush only cleared namespace a

    def test_keys_strips_namespace(self):
        t = tier()
        t.set("user:1", "a")
        t.set("user:2", "b")
        t.set("session:1", "c")
        self.assertEqual(sorted(t.keys("user:*")), ["user:1", "user:2"])
        self.assertEqual(len(t.keys()), 3)

    def test_ttl_semantics(self):
        t = tier()
        self.assertEqual(t.ttl("missing"), -2)
        t.set("k", "v")
        self.assertEqual(t.ttl("k"), -1)
        t.set("s", "v", ttl_ms=5000)
        self.assertTrue(0 < t.ttl("s") <= 5000)


class TestTieredMemoryWithRedis(unittest.TestCase):
    def test_global_tier_is_swappable_and_locals_stay_in_process(self):
        shared = RedisMemoryTier(client=FakeRedis(), namespace="fleet")
        memory = TieredMemory(shared=shared)
        a = memory.binding_for("worker-0")
        b = memory.binding_for("worker-1")

        # Same global tier for both workers; private locals.
        a.shared.set("budget", "5")
        self.assertEqual(b.shared.get("budget"), "5")
        a.local.set("secret", "a-only")
        self.assertIsNone(b.local.get("secret"))

    def test_guest_uses_redis_backed_shared_tier_unchanged(self):
        shared = RedisMemoryTier(client=FakeRedis(), namespace="fleet")
        memory = TieredMemory(shared=shared)
        box = Sandbox.create(policy={"cold_start_ms": 0}, memory=memory.binding_for("w0"))
        # Identical guest code as the in-process case — only the tier changed.
        result = box.run("return ctx.shared.incr('requests')")
        self.assertEqual(result.value, 1)
        self.assertEqual(shared.get("requests"), "1")


if __name__ == "__main__":
    unittest.main()
