# B2 — τ adaptativo por qindex (calibração offline, H9a)

**Data:** 2026-07-20  
**Split:** validação + teste held-out — 48000 superblocos.
**Reprodução:** `python src/scripts/partition_model/b2_tau_per_qindex.py`

Compara, no risco ponderado por *regret* do crivo, o τ GLOBAL (um só para todos os CQ, política atual) contra o τ ESTRATIFICADO-ÓTIMO (cada qindex escolhe o seu; Pareto sobre todas as combinações).

## 1. `reg_frac %` a `cost_red` casado — global vs estratificado

| cost_red alvo | τ global reg_frac | τ estratificado reg_frac | ganho (pp) |
|--:|--:|--:|--:|
| 15% | 0.0002 | 0.0001 | +0.0001 |
| 20% | 0.0010 | 0.0005 | +0.0005 |
| 25% | 0.0032 | 0.0012 | +0.0019 |
| 30% | 0.0061 | 0.0028 | +0.0033 |
| 35% | 0.0117 | 0.0071 | +0.0047 |

## 2. τ por qindex que a política ótima escolhe (evidência do B2)

- **cost_red ≈ 20%** (ponto 20.2%): τ = {'cq20': 0.7, 'cq32': 0.9, 'cq43': 0.97, 'cq55': 0.95}
- **cost_red ≈ 30%** (ponto 30.1%): τ = {'cq20': 0.6, 'cq32': 0.75, 'cq43': 0.8, 'cq55': 0.9}

## 3. Fronteira por qindex isolado (τ para atingir cada cost_red)

| qindex | cr 15% | cr 20% | cr 25% | cr 30% | cr 35% |
|---|--:|--:|--:|--:|--:|
| cq20 | 0.64 | — | — | — | — |
| cq32 | 0.95 | 0.91 | 0.86 | 0.79 | 0.69 |
| cq43 | 0.98 | 0.97 | 0.96 | 0.94 | 0.88 |
| cq55 | 0.98 | 0.97 | 0.96 | 0.94 | 0.91 |

## 4. Veredito

Ganho máximo do τ estratificado sobre o global: **0.0047 pp** de reg_frac.

(reg_frac é % de sobrecarga RD; a família aprendida opera na casa de 0,00X, então interpretar o ganho em termos RELATIVOS e lembrar que a confirmação real exige encodes — o lado C já lê τ do ambiente.)
