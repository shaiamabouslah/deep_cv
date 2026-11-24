"""Static registry of monitored sites (Tunisian dams), for the map view.

Coordinates were originally approximate (public knowledge, for demo
purposes). On 2026-11-21 they were corrected for 4 of the 6 sites using a
largest-connected-component search on JRC Global Surface Water (see
reports/validation_vs_official.md): the true reservoir centroid, found
within a 15 km window around the original guess, replaces the hand-picked
point, and `buffer_m` is sized from the real measured water extent instead
of a flat 6 km guess.

Sidi El Barrak and Mellègue needed a second pass (2026-11-24): the first
search gave an implausible result for both (283 km^2 for Sidi El Barrak —
the search area merging with the nearby Mediterranean coastline; 11 ha for
Mellègue — missing the real reservoir, outside the 15 km window). Fixed by
(a) clipping the search region to land only (Tunisia + Algeria, since
Mellègue straddles the border) via USDOS/LSIB_SIMPLE/2017, which removes
the sea-merge problem, and (b) widening Mellègue's search window to 35 km.
Both now give plausible hectare values for sizeable reservoirs — see the
validation report for the unofficial plausibility check (no official
ONAGRI/DGRE figure exists for either site to validate against directly,
unlike Sidi Salem/Nebhana)."""

TUNISIA_DAMS = [
    {"id": "sidi-salem", "name": "Sidi Salem", "lat": 36.64293086592259, "lon": 9.364769393985087, "buffer_m": 6000},
    {"id": "sidi-el-barrak", "name": "Sidi El Barrak", "lat": 37.00260547898608, "lon": 9.022309459216814, "buffer_m": 6000},
    {"id": "nebhana", "name": "Nebhana", "lat": 35.927962321042685, "lon": 9.710216588280908, "buffer_m": 1500},
    {"id": "bir-mcherga", "name": "Bir M'cherga", "lat": 36.49083245930035, "lon": 9.995857745945944, "buffer_m": 2200},
    {"id": "mellegue", "name": "Mellègue", "lat": 36.29642693505136, "lon": 8.66845929885142, "buffer_m": 3000},
    {"id": "lakhmes", "name": "Lakhmes", "lat": 35.99533500122221, "lon": 9.479957092085588, "buffer_m": 1500},
]
