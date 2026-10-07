# 2026-10-07 — decision head: the whole episode in context

Owner: "history is vital for a model to learn in context". Until now the decision head saw the
manual, the task, the last 6 (action, feedback) pairs as one JSON blob, and only the newest images —
no earlier frames, no turn older than 6.

`decisions.state_layout: transcript` puts the whole episode in `state` as append-only parts:
manual, task, every step's action and feedback in order, past turns' images every
`history_image_every` turns, and the current images last. The default stays `window` (runs stay
comparable); configs opt in.

Cache probe (3 calls, mock frames, transcript layout, 7 turns, every image): identical request
twice → 6170 input tokens, $0.000617 both times, 0.84 / 0.78 s; one more turn → 6377 tokens,
$0.000638, 0.80 s. No cache fields, no cache headers: the Decisions API does not cache, so history
is billed every turn — cheap ($0.10 / M input; ~200 tokens per turn per 448 px image; a 100-turn
episode with every image ≈ 26k tokens ≈ $0.0026 per call at the end), latency flat so far.
Not measured yet: whether history improves control (an A/B: window vs transcript, same seeds).
