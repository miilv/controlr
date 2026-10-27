"""dojo.perception.catalog: the MDL preset colour reader (no pxr, no assets)."""
import numpy as np
import pytest

from dojo.perception.catalog import mdl_preset_color

# The shape of a vMaterials file (ABS_Soft_Leather.mdl, the stacking toy's pieces): a base material with a colour
# parameter and named presets that set it; an OmniPBR preset with a diffuse texture (copper, marble).
MDL = """
export material ABS_Soft_Leather(
    uniform bool infinite_tiling = false [[ ::anno::display_name("Infinite Tiling") ]],
    color color_2 = color( 0.012983f, 0.012983f, 0.012983f) [[
        ::anno::description("Choose the color of the leather."),
        ::anno::display_name("Color")
    ]]
) = let { color c = color_2; } in material();

export material ABS_Soft_Leather_Pop_Orange(*)
[[
    ::anno::display_name("ABS Soft Leather Pop Orange"),
    ::anno::key_words(string[]("orange", "bright"))
]]
= ABS_Soft_Leather
(
    infinite_tiling: false,
    color_2: color(0.783538f, 0.412543f, 0.015996f),  // the leather
    texture_translate: float2(0.0f),
    texture_scale: float2(1.f),
    uv_space_index: 0
);

export material ABS_Soft_Leather_Grey(*)
= ABS_Soft_Leather(color_2: color(0.5f));

export material Copper(*)
[[ ::anno::display_name("Omni PBR ") ]]
 = ::OmniPBR::OmniPBR(
    diffuse_color_constant: color(0.200000003f, 0.200000003f, 0.200000003f),
    diffuse_texture: texture_2d("../textures/Copper/Copper_BaseColor.png" /* tag 3213 */, ::tex::gamma_srgb),
    emissive_color: color(1.f, 0.100000001f, 0.100000001f)
);

export material Glow(*)
 = ::OmniPBR::OmniPBR(
    diffuse_texture: texture_2d(),
    emissive_color: color(1.f, 0.1f, 0.1f),
    diffuse_color_constant: color(0.0f, 0.5f, 1.0f)
);
"""


def _srgb(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def test_preset_colour_is_read_and_converted_to_srgb():
    c = mdl_preset_color(MDL, "ABS_Soft_Leather_Pop_Orange")
    assert c == pytest.approx(_srgb([0.783538, 0.412543, 0.015996]), abs=1e-6)
    assert c == pytest.approx([0.898, 0.675, 0.133], abs=2e-3)       # the yellow-orange the renderer shows
    assert mdl_preset_color(MDL, "ABS_Soft_Leather_Grey") == pytest.approx(_srgb([0.5] * 3), abs=1e-6)


def test_texture_driven_presets_and_unknown_names_keep_the_default():
    assert mdl_preset_color(MDL, "Copper") is None                   # the texture is the colour, not the constant
    assert mdl_preset_color(MDL, "ABS_Soft_Leather") is None         # the base material, not a preset
    assert mdl_preset_color(MDL, "Nope") is None


def test_empty_texture_slot_and_non_surface_colours():
    # an empty diffuse_texture does not count as a texture; the emissive colour is not the surface colour
    assert mdl_preset_color(MDL, "Glow") == pytest.approx(_srgb([0.0, 0.5, 1.0]), abs=1e-6)
