# Task Queue Graph IA Brief

## Page

`task-queue-graph.html`

## Goal

Make the task queue graph open on the current section/task first instead of exposing all sections as equal-weight choices.

## Primary User Task

Open the graph and understand the recommended task section in under 15 seconds.

## First-Screen Failures To Correct

- Query context exists, but the default landing section still behaves like a generic last-section view.
- All sections compete in the selector with the same weight.
- The page does not explain clearly whether the current section is the focus section or only a reference section.
- When task/node focus misses, the fallback to the canonical queue section is not explicit enough.

## Structured Sources

- `scope-library.json`
- `graph-library.json`

## Important Objects

- `taskIndex`
- `nodeId`
- `scopeId`

## Information Architecture

### Current Context

Keep the current query context visible at the top:

- target task
- target node
- current section
- focus mode vs reference mode

### Recommended Entry

This block should answer:

- why this section is shown first
- whether the task hit is direct or best-effort
- when to fall back to the canonical queue section

### Current Section Focus

This is the main queue graph:

- the best matching section when there is a query
- a stable current section when there is no query
- visually emphasize the target task and its local neighborhood

### Other Sections

Other sections stay available, but they must step down:

- keep selector access
- label focus vs reference sections
- do not let all sections feel equally important on first open

## Large-Data Behavior

- default to focused section exploration
- highlight the target task first
- dim unrelated tasks when focus is active
- keep other sections available through selector navigation instead of equal-weight first-screen exposure

## Do Not Change

- `taskIndex` / `nodeId` semantics
- query parameter names
- canonical report anchor contract
- section data meaning
