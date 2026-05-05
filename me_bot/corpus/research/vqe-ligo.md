---
title: "VQE × LIGO classifier"
slug: vqe-ligo
visibility: public
last_reviewed: 2026-05-04
tags: [research, quantum, gravitational-waves, ligo, preprint]
---

# VQE × LIGO classifier

## What it is

A variational quantum eigensolver classifier trained to discriminate
true binary-merger gravitational-wave signals from glitch artefacts in
LIGO O3 strain data. The classifier runs on IBM Quantum's
superconducting `ibm_nairobi` processor — a real quantum backend, not
a simulator — and on PennyLane with the lightning.qubit local
simulator for development.

## What's novel

Most published work on quantum machine learning for gravitational
waves runs entirely on classical simulators. Running on real
superconducting hardware exposes the classifier to noise, decoherence,
and gate-level error that simulator work doesn't capture. The
preprint compares simulated and on-hardware performance directly.

The architecture also uses physics-motivated encoding choices —
amplitude encoding for the strain windows after whitening — rather
than treating the input as an arbitrary feature vector.

## Status

The preprint is in active preparation as of 2026. The collaboration
is with a UCSB PhD candidate; Tardigrade Innovation LLC is the
contracting party for Raven's contributions. The work is being
developed in the orbit of the LIGO Scientific Collaboration but is
not yet a formal LSC publication.

## What it builds toward

The same architecture has natural extensions toward exotic compact
object detection (sub-threshold signals from boson stars and
non-standard compact remnants outside the matched-filter template
banks) and binary parameter estimation. The Clifford geometric algebra
work in `tardigrade_agent` is a complementary architectural direction
that may eventually be combined with the variational classifier.

## Public artifacts

- arXiv preprint (forthcoming)
- Code release alongside the preprint
- Cleaned strain dataset (subset, for educational use)
- Live demo on the Tardigrade site
