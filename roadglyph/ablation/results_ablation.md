## RoadGlyph Ablation Study Results

| Tag | Description | DS | SR | Eff | Cmf | Δ_spd | Δ_rt | Δ_sum |
|-----|-------------|----|----|-----|-----|-------|------|-------|
| **Full** | inference Δ_wp | — | — | — | — | 1.020 | 0.364 | 1.384 |
| **A1** | γ=1, β=0 forced | — | — | — | — | 0.000 | 0.000 | 0.000 |
| **A2** | e_lat=e_lon=0 | — | — | — | — | 0.000 | 0.000 | 0.000 |
| **A1*** | γ=1, β=0 (heads/losses retained) | — | — | — | — | — | — | — |
| **A2*** | e_lat, e_lon=0 | — | — | — | — | — | — | — |
| **B1** | Pass-1 removed, lon head uses v_pool directly | — | — | — | — | — | — | — |
| **B2** | lon head input e_ctx → v_pool | — | — | — | — | — | — | — |
| **C1** | teacher forcing fully disabled | — | — | — | — | — | — | — |
| **C2** | detach removed for samples without labels | — | — | — | — | — | — | — |

**DS**: Driving Score, **SR**: Success Rate, **Eff**: Efficiency, **Cmf**: Comfortness
**Δ_spd/rt/sum**: Token-intervention consistency (larger = more sensitive to token change)

*1 seed (seed=42). Δ_wp: N=256 val frames, all (lat×lon)=(4×8) intervention pairs.*
