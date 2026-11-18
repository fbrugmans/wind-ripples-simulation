"""The run series of the article: complete settings per series (passed to the model code as environment variables).

Each series was run with seeds 271, 272, 273 and 274. The series name is also the suffix of the output file
(c23_z<seed>_<series>.npz). SR0 is the article model; the other series are the variants compared in the article
(Tables 3, 4, 7 and 8 and the sections on the wind shadow and initiation).
"""

# Common grain and bed rules of all series.
_CORE = dict(CVAR="C23", KORRELMASSA="1", VLAGEN="0", SPLASH_FAC="1.0", WAND="3", CEL_TOREN="tan", TILT_PLUS="1",
             HOEK_LSQ_KOL="12", PCR="0.2116", WINDROL="2", WR_ZETA="1000000000.0", TEIND="14400")

# Earlier model versions (fixed A and B for the wind, creep distance scaled with the local impact rate).
_EARLY = dict(_CORE, KRUIP_INSLAG="1")
_MIRROR = dict(_EARLY, KRUIP_TERUG="1", KAATS_TERUG="1")      # + creep and rebound in the direction of the impacting grain

# Later versions: creep distance not scaled with the impact rate.
_LATE = dict(_CORE, KRUIP_INSLAG="0", KRUIP_TERUG="1", KAATS_TERUG="1", MOD_UB="0.20", MOD_KROM="7")
_VISCOUS = dict(_LATE, MOD_AB="2")                            # wind with viscous sublayer (van Driest) and curvature correction
_ROUGH = dict(_LATE, MOD_AB="1", MOD_Z0="0.1")                # wind from the theory for a fully rough bed, z0 = 0.1 mm
ARTICLE = dict(_ROUGH, STROOM="1")                            # + streamlines follow the bed: the article model

SHADOW = dict(OMH_GR="5.0")                                   # imposed wind shadow: straight line 5 degrees below the horizontal

SERIES = {
    # the article model
    "SR0": ARTICLE,
    "SL16": dict(ARTICLE, TEIND="57600"),                     # the same, 16 hours
    "SG0": dict(ARTICLE, WINDROL="0"),                        # without wind-driven sliding
    "SR5": dict(ARTICLE, **SHADOW),
    # rough bed, without streamlines
    "SQ0": _ROUGH, "SQ5": dict(_ROUGH, **SHADOW), "SQ8": dict(_ROUGH, OMH_GR="8.0"), "SQ3": dict(_ROUGH, OMH_GR="3.0"),
    # viscous sublayer and curvature correction
    "PKO": _VISCOUS, "PKS": dict(_VISCOUS, **SHADOW),
    "CO10": dict(_VISCOUS, COHESIE="0.1"), "CO25": dict(_VISCOUS, COHESIE="0.25"), "CO50": dict(_VISCOUS, COHESIE="0.5"),
    # earlier versions (creep distance scaled with the local impact rate)
    "NMO": _EARLY, "NMS": dict(_EARLY, **SHADOW),                                   # fixed A = 3, B = 1
    "KAO": _MIRROR, "KAS": dict(_MIRROR, **SHADOW),                                 # + mirrored rebound
    "ABO": dict(_MIRROR, MOD_AB="1"), "ABS": dict(_MIRROR, MOD_AB="1", **SHADOW),   # rough bed, z0 of sand at rest (d/30)
    "AZO": dict(_MIRROR, MOD_AB="1", MOD_Z0="0.1"), "AZS": dict(_MIRROR, MOD_AB="1", MOD_Z0="0.1", **SHADOW),   # rough bed, z0 0.1 mm
    "VLO": dict(_MIRROR, MOD_AB="2", MOD_UB="0.20"), "VLS": dict(_MIRROR, MOD_AB="2", MOD_UB="0.20", **SHADOW), # viscous sublayer
    "BRLO": dict(_MIRROR, MOD_AB="2", MOD_UB="0.20", MOD_KROM="7"),                  # viscous sublayer + curvature correction
    "BRLS": dict(_MIRROR, MOD_AB="2", MOD_UB="0.20", MOD_KROM="7", **SHADOW),
}

# Initiation (Table 3 and the imposed ripplets in the Discussion): the article model at u* 0.31 and 0.50 m/s,
# K = first 120 s with a record every second, L = 30 minutes with a record every 5 s;
# RN flat bed, RA ripplet of 35 mm (61 waves in the bed), RB ripplet of 102 mm (21 waves), two grains high.
for _u in ("31", "50"):
    for _k, _sin in (("RN", None), ("RA", "61,0.2"), ("RB", "21,0.2")):
        for _len, _t, _dt in (("K", "120", "1"), ("L", "1800", "5")):
            _s = dict(ARTICLE, USTAR=f"0.{_u}", TEIND=_t, OPNAME_DT=_dt)
            if _sin: _s["BED_SINUS"] = _sin
            SERIES[f"{_k}{_u}{_len}"] = _s

SEEDS = (271, 272, 273, 274)
