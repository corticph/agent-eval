---
status: accepted
---

# Two-level agent attribution

With per-step agent overrides (AGENT-986), a sequential case can switch agents mid-context, so "the agent this case talked to" is no longer a single value. We decided that `EvaluationResult.agent_id` always means the **case agent** — the case's default, one fixed meaning — and that per-step attribution lives solely on `StepResult.agent_id`, which names the agent that actually messaged that step (including a harness failure's synthetic trail row, which blames the step agent of the step in flight). We considered attributing both fields to the actual agent in flight, but rejected it: a case-level field that sometimes means "default" and sometimes "last-touched" gives JSON consumers no way to tell which they are reading.

## Consequences

- Reporting renders `EvaluationResult.agent_id` as the default agent; a report reader checking which agent executed a given step reads the Step Trail, not the case bullet.
- Sinks that stamp step rows must use the step's own `agent_id`, not the case-level one.
- Timeouts attribute at the Case Agent level because the parent thread cannot reliably identify the in-flight step.
- Flipping this later means changing the result payload schema and every sink that consumes it, hence the record.
