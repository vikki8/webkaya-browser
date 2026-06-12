import unittest

from webkaya import EbpfEnv, EbpfMap, EbpfVm, asm, verify_program
from webkaya import asm as a


def run(chunks, ctx=b"", env=None):
    return EbpfVm(a.assemble(chunks)).run(ctx, env)


class TestEbpfArithmetic(unittest.TestCase):
    def test_immediate_and_register_math(self):
        self.assertEqual(run([a.mov_imm(0, 40), a.add_imm(0, 2), a.exit_()]), 42)
        self.assertEqual(run([a.mov_imm(0, 6), a.mul_imm(0, 7), a.exit_()]), 42)
        self.assertEqual(
            run([a.mov_imm(1, 30), a.mov_imm(2, 12), a.mov_reg(0, 1), a.add_reg(0, 2), a.exit_()]),
            42,
        )

    def test_lddw(self):
        self.assertEqual(run([a.lddw(0, 0xDEADBEEFCAFEBA), a.exit_()]), 0xDEADBEEFCAFEBA)

    def test_conditional_branches(self):
        def clamp_at_10(value):
            return run([a.mov_imm(0, value), a.jgt_imm(0, 10, 1), a.ja(1), a.mov_imm(0, 10), a.exit_()])

        self.assertEqual(clamp_at_10(3), 3)
        self.assertEqual(clamp_at_10(30), 10)

    def test_div_mod_by_zero(self):
        self.assertEqual(run([a.mov_imm(0, 5), a.mov_imm(2, 0), a.div_reg(0, 2), a.exit_()]), 0)
        self.assertEqual(run([a.mov_imm(0, 5), a.mov_imm(2, 0), a.mod_reg(0, 2), a.exit_()]), 5)

    def test_alu32_zero_extends(self):
        self.assertEqual(run([a.mov32_imm(0, -1), a.exit_()]), 0xFFFFFFFF)

    def test_instruction_limit(self):
        vm = EbpfVm(a.assemble([a.ja(-1), a.exit_()]), max_instructions=1000)
        with self.assertRaisesRegex(ValueError, "instruction limit"):
            vm.run(b"")


class TestEbpfMemory(unittest.TestCase):
    def test_reads_context(self):
        import struct
        ctx = struct.pack("<QQ", 7, 5)
        self.assertEqual(
            run([a.ldxdw(0, 1, 0), a.ldxdw(2, 1, 8), a.add_reg(0, 2), a.exit_()], ctx), 12
        )

    def test_stack_round_trip(self):
        self.assertEqual(
            run([a.mov_imm(1, 42), a.stxdw(10, -8, 1), a.ldxdw(0, 10, -8), a.exit_()]), 42
        )

    def test_context_is_read_only(self):
        ctx = b"\x00" * 8
        with self.assertRaisesRegex(ValueError, "read-only"):
            run([a.mov_imm(2, 1), a.stxdw(1, 0, 2), a.exit_()], ctx)

    def test_out_of_bounds(self):
        ctx = b"\x00" * 16
        with self.assertRaisesRegex(ValueError, "out-of-bounds"):
            run([a.ldxdw(0, 1, 64), a.exit_()], ctx)


class TestEbpfHelpers(unittest.TestCase):
    def test_trace(self):
        traced = []
        run(
            [a.mov_imm(1, 123), a.call(asm.TRACE), a.mov_imm(0, 0), a.exit_()],
            env=EbpfEnv(trace=traced.append),
        )
        self.assertEqual(traced, [123])

    def test_map_round_trip(self):
        m = EbpfMap()
        result = run(
            [
                a.mov_imm(1, 0), a.mov_imm(2, 5), a.mov_imm(3, 77), a.call(asm.MAP_SET),
                a.mov_imm(1, 0), a.mov_imm(2, 5), a.call(asm.MAP_GET), a.exit_(),
            ],
            env=EbpfEnv(maps=[m]),
        )
        self.assertEqual(result, 77)
        self.assertEqual(m.get(5), 77)

    def test_map_add_accumulates(self):
        m = EbpfMap()
        vm = EbpfVm(a.assemble([a.mov_imm(1, 0), a.mov_imm(2, 0), a.mov_imm(3, 1),
                              a.call(asm.MAP_ADD), a.mov_imm(0, 0), a.exit_()]))
        vm.run(b"", EbpfEnv(maps=[m]))
        vm.run(b"", EbpfEnv(maps=[m]))
        self.assertEqual(m.get(0), 2)


class TestVerifier(unittest.TestCase):
    def test_rejects_empty_and_misaligned(self):
        with self.assertRaises(ValueError):
            verify_program(b"")
        with self.assertRaises(ValueError):
            verify_program(b"\x00" * 7)

    def test_requires_exit(self):
        with self.assertRaisesRegex(ValueError, "end with exit"):
            verify_program(a.assemble([a.mov_imm(0, 0)]))

    def test_rejects_out_of_range_jump(self):
        with self.assertRaisesRegex(ValueError, "invalid target"):
            verify_program(a.assemble([a.ja(5), a.exit_()]))

    def test_rejects_write_to_r10(self):
        with self.assertRaisesRegex(ValueError, "read-only r10"):
            verify_program(a.assemble([a.mov_imm(10, 1), a.exit_()]))

    def test_rejects_unknown_helper(self):
        with self.assertRaisesRegex(ValueError, "unknown helper"):
            verify_program(a.assemble([a.call(99), a.exit_()]))


if __name__ == "__main__":
    unittest.main()
