/**
 * Minimal eBPF instruction builder for authoring probe programs in tests
 * and host code without a separate toolchain. Emits standard eBPF encoding,
 * so programs assembled here also load into kernel/ubpf runtimes.
 */

export function insn(opcode: number, dst: number, src: number, offset: number, imm: number): Uint8Array {
  const out = new Uint8Array(8);
  const view = new DataView(out.buffer);
  view.setUint8(0, opcode);
  view.setUint8(1, ((src & 0x0f) << 4) | (dst & 0x0f));
  view.setInt16(2, offset, true);
  view.setInt32(4, imm | 0, true);
  return out;
}

export const op = {
  // ALU64
  movImm: (dst: number, imm: number) => insn(0xb7, dst, 0, 0, imm),
  movReg: (dst: number, src: number) => insn(0xbf, dst, src, 0, 0),
  addImm: (dst: number, imm: number) => insn(0x07, dst, 0, 0, imm),
  addReg: (dst: number, src: number) => insn(0x0f, dst, src, 0, 0),
  subImm: (dst: number, imm: number) => insn(0x17, dst, 0, 0, imm),
  mulImm: (dst: number, imm: number) => insn(0x27, dst, 0, 0, imm),
  divReg: (dst: number, src: number) => insn(0x3f, dst, src, 0, 0),
  modReg: (dst: number, src: number) => insn(0x9f, dst, src, 0, 0),
  andImm: (dst: number, imm: number) => insn(0x57, dst, 0, 0, imm),
  // ALU32
  mov32Imm: (dst: number, imm: number) => insn(0xb4, dst, 0, 0, imm),
  // 64-bit immediate load (two slots)
  lddw: (dst: number, value: bigint): Uint8Array => {
    const v = BigInt.asUintN(64, value);
    const out = new Uint8Array(16);
    out.set(insn(0x18, dst, 0, 0, Number(v & 0xffffffffn) | 0), 0);
    out.set(insn(0x00, 0, 0, 0, Number(v >> 32n) | 0), 8);
    return out;
  },
  // Memory
  ldxdw: (dst: number, src: number, offset: number) => insn(0x79, dst, src, offset, 0),
  ldxw: (dst: number, src: number, offset: number) => insn(0x61, dst, src, offset, 0),
  stxdw: (dst: number, offset: number, src: number) => insn(0x7b, dst, src, offset, 0),
  stdw: (dst: number, offset: number, imm: number) => insn(0x7a, dst, 0, offset, imm),
  // Jumps
  ja: (offset: number) => insn(0x05, 0, 0, offset, 0),
  jeqImm: (dst: number, imm: number, offset: number) => insn(0x15, dst, 0, offset, imm),
  jneImm: (dst: number, imm: number, offset: number) => insn(0x55, dst, 0, offset, imm),
  jgtImm: (dst: number, imm: number, offset: number) => insn(0x25, dst, 0, offset, imm),
  jgeImm: (dst: number, imm: number, offset: number) => insn(0x35, dst, 0, offset, imm),
  jltImm: (dst: number, imm: number, offset: number) => insn(0xa5, dst, 0, offset, imm),
  jleImm: (dst: number, imm: number, offset: number) => insn(0xb5, dst, 0, offset, imm),
  jsgtImm: (dst: number, imm: number, offset: number) => insn(0x65, dst, 0, offset, imm),
  jeqReg: (dst: number, src: number, offset: number) => insn(0x1d, dst, src, offset, 0),
  jgtReg: (dst: number, src: number, offset: number) => insn(0x2d, dst, src, offset, 0),
  // Control
  call: (helperId: number) => insn(0x85, 0, 0, 0, helperId),
  exit: () => insn(0x95, 0, 0, 0, 0),
};

export function assemble(chunks: Uint8Array[]): Uint8Array {
  const total = chunks.reduce((sum, c) => sum + c.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}
