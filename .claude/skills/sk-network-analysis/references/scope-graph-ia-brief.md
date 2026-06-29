# Scope Graph IA Brief

## Page

`scope-graph.html`

## Goal

Make the scope graph land users on the current object first instead of treating every round as equal-weight exploration.

## Primary User Task

Open the graph and understand the recommended scope/node context in under 15 seconds.

## First-Screen Failures To Correct

- Query focus exists, but the page still feels like a full-round browser first.
- Repeated rounds compete with the current round instead of stepping down into reference status.
- Users are not told clearly whether the current round is a direct object hit or only a reference round.
- When query focus misses, the fallback to the canonical scope section is not explicit enough.

## Structured Sources

- `scope-library.json`
- `graph-library.json`

## Important Objects

- `scopeId`
- `nodeId`
- `streamId`

## Information Architecture

### Current Context

Keep the current query context visible at the top:

- target scope
- target node
- current round
- focus mode vs reference mode

### Recommended Entry

This block should answer:

- why this round is being shown first
- whether the object hit is direct or best-effort
- when to fall back to the canonical scope section

### Current Round Focus

This is the main graph view:

- canonical round when there is no query
- best matching round when there is a query
- emphasize the target scope and nearby nodes

### Reference Rounds

Other rounds are still useful, but they must step down:

- keep tab access
- visually mark them as reference
- do not let repeated rounds dominate the first screen

## Large-Data Behavior

- default to focused exploration, not full-round scanning
- highlight one primary object and one nearby scope neighborhood
- dim unrelated nodes when focus is active
- treat duplicate/reference rounds as secondary navigation

## Do Not Change

- `scopeId` / `nodeId` / `streamId` semantics
- query parameter names
- canonical report anchor contract
- round data meaning
