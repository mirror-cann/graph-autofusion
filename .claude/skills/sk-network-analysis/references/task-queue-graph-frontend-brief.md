# Task Queue Graph Frontend Brief

## Page

`task-queue-graph.html`

## Visual Direction

A queue operations console with one active section in focus:

- structured, industrial layout
- visible current-section emphasis
- supporting sections present, but clearly secondary
- task-level emphasis through opacity and stroke, not noisy chrome

## Typography

- compact operational header
- neutral system-like body text
- monospace for task identifiers and queue metadata

## Component Rules

### Focus Summary

- must act like a routing hint, not a generic description block
- clearly state:
  - current focus section
  - current task or node context
  - fallback when there is no direct hit

### Section Selector

- current focus section should read as the primary working section
- other sections should remain accessible, but clearly feel like reference alternatives

### Queue Graph

- preserve current SVG rendering and tooltips
- visually prioritize the target task and its local neighborhood
- dim unrelated tasks when focus mode is active

## Interaction Rules

- preserve current zoom, pan, keyboard, and tooltip behavior
- use low-friction highlighting only
- avoid strong animation or decorative motion

## Large-Data Behavior

- first screen should feel like a focused section review, not a queue dump
- keep other sections behind selector navigation
- current section and current task must remain obvious without scanning the whole canvas

## Do Not Change

- queue graph semantics
- existing query parameter contract
- canonical report fallback links
