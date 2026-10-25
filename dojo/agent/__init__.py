"""dojo.agent: the hybrid LLM agent layer (skills or code cells over a ``robo`` API).

* ``llmio``      omniroute client factory (OMNIROUTE_* only), per-call accounting;
* ``manual``     the agent's operating manual: skill library, primitive API, checker conventions;
* ``scene``      WorldState -> the scene text the agent reads (neutral object ids);
* ``grounding``  Phase 1 benchmark: instruction + scene -> skill choice / code, per model.
"""
