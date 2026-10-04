"""RoboDojo backend: controlr drives RoboDojo's dual ARX X5 through RoboDojo's own eval client.

``serve.py`` (controlr side, ``controlr robodojo-serve``) <-> ``protocol.py`` <-> ``shim/deploy.py``
(inside RoboDojo's eval client, installed as ``XPolicyLab/policy/controlr``). See
``docs/ROBODOJO.md``.
"""
