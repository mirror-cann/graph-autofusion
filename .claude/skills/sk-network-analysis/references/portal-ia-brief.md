# Run Portal IA Brief

## Page

`run-portal.html`

## Goal

Make the portal the first diagnosis cockpit instead of a flat report directory.

## Primary User Task

Decide what to read next in under 30 seconds.

## First-Screen Failures To Correct

- Too many equal-weight cards still compete on the first screen.
- `graph-capable` vs `summary-only` is visible, but not dominant enough.
- Primary vs secondary entry exists, but the jump path is still too directory-like.
- Large runs such as `dual_sk` still create too much first-screen scanning pressure.

## Structured Sources

- `run-portal.html`
- `scope-library.json`
- `graph-library.json`

## Important Objects

- `scopeId`
- `nodeId`
- `taskIndex`
- `streamId`

## Information Architecture

### Hero Summary

Keep only:

- run id
- modelRI
- diagnostic completeness
- capability mode
- presentation mode
- primary next step

### Current Decision

This is the first card users must read. It must answer:

- what mode this run is in
- what the top actionable signal is
- what the primary next step is

### Primary Evidence

Keep only the evidence that helps users trust the next move:

- validation
- input type
- event/log counts
- object-key counts

### Current Capability

Show ability-level chips, not file lists.

- available capabilities
- still missing capabilities

### What To Read Next

This is the decision hub. Each item must include:

- target page
- target object
- reason to open it now

Keep this list short and ranked.

### Navigation Layers

- `Primary Entry`
  - at most 3 cards
  - should bias toward graph or update when `graph-capable`
  - should bias toward summary guidance when `summary-only`
- `Secondary Views`
  - useful but not first read
- `Reference Views`
  - structured/supporting pages

### Folded Areas

Default to folded:

- asset status directory/file detail
- node trace detail
- secondary report catalogs
- asset guidance detail

## Large-Data Behavior

- Default to `focused`
- Never let the first screen become a report index
- Use `Top-N + fold + aggregate`
- If there are many valid entries, keep one primary path and at most two backups

## Do Not Change

- `capability_mode` semantics
- `presentation_mode` semantics
- `diagnostic_completeness`
- `current_capabilities`
- `still_missing_capabilities`
- any JSON field names or diagnostic meaning
