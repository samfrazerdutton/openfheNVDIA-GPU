# Device Residency Design

Status: design, no implementation yet. This document exists because the two
prior residency attempts crashed, and both were design errors written into
code before the design was settled — a deferred-writeback segfault in decrypt,
and stale device data served because residency was inferred from a recycled
host pointer. This document fixes the design first.

## Goal

Keep a ciphertext's polynomial data resident in VRAM across an evaluation
circuit (NTT -> multiply -> keyswitch -> rescale -> ...) so operands cross
PCIe once per circuit instead of once per operation. The paired-difference
sweep (RESULTS_paired_sweep.md) proved operation-level dispatch cannot beat
CPU on this hardware because per-op transfer scales with data volume; residency
is the only remaining path to a real speedup.

## What the OpenFHE structure forces

Four facts about DCRTPolyImpl / PolyImpl determine the whole design. None of
this is a free choice; the code shape dictates it.

1. Clone and copy DEEP-COPY m_vectors (dcrtpoly.h copy-ctor and assignment).
   A cloned ciphertext owns independent tower buffers. It must never share a
   device slot with its source.

2. Rescale mutates IN PLACE via m_vectors.resize(size-1) (DropLastElement).
   Dropping a tower does not move or reallocate the surviving towers' buffers.
   Therefore a key that includes tower count or spans the whole ciphertext
   would invalidate on every rescale even though the surviving data is
   byte-identical.

3. DCRTPoly has no own SetFormat; it delegates to per-tower PolyImpl::SetFormat.
   The COEFFICIENT<->EVALUATION transform — the operation we route to the GPU
   NTT — happens per tower.

Conclusion: residency is PER TOWER, and the hook that runs the transform and
the cache that owns the device copy are at the SAME granularity (PolyImpl).

## The key

Cache key = the tower's coefficient-buffer data pointer: PolyImpl::m_values->data().

Why this and not the alternatives:
- Not the host buffer at the HAL boundary: OpenFHE recycles those addresses,
  which is exactly what served stale data before.
- Not the DCRTPoly object address: rescale resizes it in place, and temporaries
  can be destroyed and reborn at the same address (ABA).
- The tower data pointer survives rescale (surviving towers don't move),
  distinguishes clones (deep copy = fresh allocation = different address), and
  is genuine data identity: two PolyImpls sharing a data pointer ARE the same
  coefficients, which is precisely when sharing a device copy is correct.

## The validity guard (fixes the stale-data crash)

A pointer match is necessary but NOT sufficient — the prior crash assumed it was.
Within a circuit a tower buffer can be freed and a new PolyImpl allocated at the
same address (ABA), and the same buffer legitimately changes format across an
NTT. So each cache entry stores:

- the data pointer (key)
- the format flag the device copy is in (COEFFICIENT or EVALUATION)
- a monotonic generation counter stamped at entry creation

Lookup rule: a hit requires pointer match AND the caller's expected format to
match the entry's recorded format. Any mismatch is treated as a miss and
triggers re-upload. The device copy is never trusted on a bare pointer match.

## The writeback rule (fixes the decrypt segfault)

The prior segfault came from deferring writeback and letting a CPU-side consumer
read a host buffer the device had silently taken ownership of. Rule: the device
copy is authoritative only while execution stays inside GPU-routed operations.
Any operation not routed through the HAL (decrypt, serialization, anything
CPU-side) must see correct host data. Therefore:

- A resident tower is written back to its host buffer before any un-hooked
  consumer can observe it.
- The safe default is eager writeback at the end of each GPU op, with
  upload-skip (don't re-upload an unchanged resident buffer) as the only
  optimization enabled first. Deferred writeback across ops is a SEPARATE,
  later optimization and must not be bundled with the first residency landing —
  that bundling is what crashed decrypt before.

## Landing order (each step independently validated)

1. Per-tower VRAM cache keyed as above, upload-skip only, eager writeback.
   Validate: full CKKS e2e still decrypts bit-correct; count uploads skipped.
2. Route PolyImpl::SetFormat(EVALUATION) through the proven GPU CT-NTT for
   resident towers. Validate against test_ntt_towers semantics in-pipeline.
3. Only then consider deferred writeback across chained ops, guarded by the
   format/generation validity check, with the paired-difference harness
   measuring whether it actually wins.

## Non-goals for the first landing

- No deferred cross-op writeback (step 3, later).
- No keyswitch residency yet (separate, after multiply residency is proven).
- No graph fusion yet (that is step 4 of the overall plan).
