# Scope Graph Frontend Brief

## Page

`scope-graph.html`

## Visual Direction

A technical graph workspace with a strong first-read focus lane:

- calm, bright canvas
- compact focus summary above the graph
- clear distinction between current round and reference rounds
- graph remains interactive, but the first screen should feel guided

## Typography

- crisp serif or editorial heading tone for page-level labels
- compact sans-serif body copy
- monospace for object identifiers and round details

## Component Rules

### Focus Summary

- compact operator-style briefing card
- must clearly show:
  - focus mode or reference mode
  - current target object
  - fallback guidance when there is no direct hit

### Tabs

- focused round should visually stand out
- reference rounds should remain usable but visually step down
- repeated rounds should not dominate the first scan

### Graph Area

- keep existing SVG interaction
- add emphasis through node opacity and stroke treatment, not through heavy decoration
- unrelated nodes should visually fade when focus mode is active

## Interaction Rules

- preserve current zoom/pan behavior
- preserve tooltip behavior
- do not introduce noisy animation
- use subtle state change to communicate focus vs reference

## Large-Data Behavior

- primary visual energy should stay on the target scope/node
- do not encourage full-graph scanning as the default reading pattern
- reference rounds should feel intentionally secondary

## Do Not Change

- graph drawing semantics
- existing query parameter contract
- canonical report fallback links
