"""Vocabulary routing for pour_liquid_into_cup(_random): the public instruction template is in the table (camera:
the catalogue fallback sent 8 / 8 _random episodes to pour_by_language)."""
from dojo.runtime import supervisor as sv


def test_the_pour_liquid_instruction_is_routed_by_the_table():
    v = sv.perception_vocabulary("Pour the liquid from the bottle into the cup.", fallback=False)
    assert v is not None and v.name == "pour_liquid_into_cup"
    by_lang = ("Pour the liquid from the first red bottle into the first black bowl, from the second violet bottle into "
               "the second white bowl, and from the third turquoise bottle into the third brown bowl. Then reset the robot arm.")
    w = sv.perception_vocabulary(by_lang, fallback=False)
    assert w is None or w.name != "pour_liquid_into_cup"
