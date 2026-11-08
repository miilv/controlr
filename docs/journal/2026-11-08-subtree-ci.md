# Subtree CI preparation

The legality ratchet now prefixes Git object paths with the project's Git prefix.
An import base without the project tree explicitly logs bootstrap; current findings
remain gated. Existing trees with missing pins fail closed. Detection-gain history
continues to support bases predating those pins. A temporary nested Git repository
tests bootstrap, pin comparisons, growth rejection and missing pins.

CI-path audit: tracked shell and secret scans are cwd-relative; file-derived project
roots remain inside the subtree. Board replay uses commit ancestry, not tree paths.
No runtime, skill, safety or default changes.
