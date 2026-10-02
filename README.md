# controlr

A robotics harness, built like a coding-agent harness: camera frames + a robot
"operating manual" system prompt + a task go to a vision LLM over an
OpenAI-compatible API, and the model replies with **low-level numeric actions**
and a status. Turn by turn, append-only, fully prompt-cached.

* Design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
* Background research: [`research/README.md`](research/README.md)

Status: v0 under construction (MuJoCo UR3 + Robotiq 2F-85 sim first; real UR3 via PHANTOM drivers later).
