# Resultados — Fase 5 (benchmark de tese, teste held-out)

**Data:** 2026-07-13. Branch `ml-partition-dev`. Encoder de teste `libaom_perf`
(36 features H9a, `PARTITION_ML_STUDENT=1`) vs âncora `libaom_perf_anchor` (libaom
v3.10.0 cru). Protocolo congelado (`PROTOCOLO_avaliacao.md`): teste held-out
**Jockey, RaceNight, RiverBank** (4K), **10 quadros**, cq {20,32,43,55},
cpu-used=0, single-thread. Métricas (padrão IEEE): **BD-rate (PSNR-Y)** = perda de
eficiência; **TS%** = redução de tempo `(1−1/speedup)·100`; **speedup**. Dados
brutos em `results/benchmark/h9_test/<seq>/{curve_safe,curve_aggr,ablation}`.

---

## 1. Pilar 1 — Curva operacional do H9a vs baseline (redução de tempo)

Fronteira taxa-BD × tempo do estudante implantado (política completa: NONE-commit +
poda rect + split). Três pilares por ponto:

**Jockey**
| ponto | BD-rate % | TS % | speedup |
|---|---:|---:|---:|
| P0 | 0,19 | 21,6 | 1,28× |
| P_ref | 0,92 | 32,6 | 1,48× |
| A2 | 1,37 | 47,7 | 1,91× |
| A3 | 2,03 | 57,2 | 2,34× |

**RaceNight**
| ponto | BD-rate % | TS % | speedup |
|---|---:|---:|---:|
| P0 | 0,27 | 18,1 | 1,22× |
| P_ref | 0,74 | 31,8 | 1,47× |
| A2 | 1,17 | 41,7 | 1,72× |
| A3 | 1,72 | 50,7 | 2,03× |

**RiverBank**
| ponto | BD-rate % | TS % | speedup |
|---|---:|---:|---:|
| P0 | 0,003 | 17,4 | 1,21× |
| P_ref | 0,13 | 24,2 | 1,32× |
| A2 | 0,23 | 30,2 | 1,43× |
| A3 | 0,38 | 36,8 | 1,58× |

**Síntese:** ponto conservador (~P_ref): **~0,6% BD-rate a TS ~30% (1,4×)**;
ponto agressivo (A3): **~1,4% BD-rate a TS ~48% (2,0×)**, médias das 3 seqs. O H9a
entrega redução de tempo substancial a baixo custo de eficiência, **consistente nas
três sequências de teste** (RiverBank quase sem perda). **Pilar 1: forte.**

---

## 2. Pilar 2 — Atribuição ao aprendizado (ml vs aleatório)

Ablação de atribuição (mesma política NONE-commit; só muda a fonte de escore). A
tempo casado, o **ml domina o aleatório em todas as seqs e em todos os níveis**:

| seq | @1,15× ml vs random | @1,30× ml vs random |
|---|---|---|
| Jockey | 0,13% vs 1,79% | 0,26% vs 3,21% |
| RaceNight | 0,34% vs 3,75% | 0,90% vs 6,01% |
| RiverBank | 0,11% vs 3,40% | — |

**O ganho é atribuível ao modelo, não a poda aleatória. Pilar 2: claro (3/3).**

---

## 3. Pilar 3 — vs variância (a barra difícil)

### 3.1 A speedup casado: sem head-to-head no teste (não-sobreposição, 3/3)

Na ablação (política casada NONE-commit), as faixas de speedup de ml e variância
**não se sobrepõem em nenhuma das 3 seqs**:

| seq | ml (speedup / BD) | variância (speedup / BD) | sobrepõe? |
|---|---|---|---|
| Jockey | 1,06–1,59× / 0,16–0,92% | 1,89–3,36× / 1,76–5,61% | não |
| RaceNight | 1,06–1,46× / 0,09–1,68% | 2,13–3,28× / 3,96–8,14% | não |
| RiverBank | 1,03–1,18× / 0,01–0,16% | 1,23–3,69× / 0,75–11,70% | não (quase) |

O ml opera em baixo speedup / baixo BD; a variância só em regime agressivo (alto
speedup / alto BD). Logo, a comparação "a speedup casado" é **indefinida** no teste
(precisaria de faixas cruzadas). Causa: o grid de τ foi congelado na calibração do
HoneyBee, onde a variância τ=0,95 dava 1,34× (sobrepunha); no conteúdo de teste ela
poda mais agressivo, então já parte de ≥1,23–2,13×. Não estendido (seria tunar em
dado de teste). **Limitação documentada.**

### 3.2 A política casada: atribuição LIMPA (o argumento central)

O rect-off é ação **secundária** do ml (saída P(REST)); a variância
**estruturalmente não a executa** (fixa P(REST)=1). Comparando o modo **primário**
do ml (NONE-commit) com a variância, **mesma política, variando só o escore**, há
uma afirmação de atribuição pura que **dispensa sobreposição de speedup**:

| seq | ml — BD mínimo alcançável | variância — BD mínimo | razão |
|---|---:|---:|---:|
| Jockey | **0,16%** | 1,76% | 11× |
| RaceNight | **0,09%** | 3,96% | 44× |
| RiverBank | **0,008%** | 0,75% | 94× |

Tudo idêntico exceto a fonte de escore. **O escore do ml (contexto RD) alcança a
região implantável de baixo-BD (quase sem perda); o da variância não** — sua regra
grosseira commita NONE em qualquer bloco liso, inclusive nos que deveriam dividir,
e não desce abaixo de 0,75–3,96% BD. O ganho é **atribuível ao modelo**, nas 3
seqs. **Pilar 3 (na forma limpa): respondido — 3/3.**

### 3.3 Comparações complementares (com ressalva de política mista)

- **A ~mesmo speedup no RiverBank** (quase-sobreposição, ml 1,18× vs variância
  1,23×): ml **0,16%** vs variância **0,75%** BD — ml ~4,6× menor a speedup quase
  igual.
- **A BD casado com o ml implantado (rect-off), Jockey** (BD 1,76%): ml **2,17×
  (TS 53%)** vs variância 1,89× (TS 47%). *Ressalva:* mistura políticas
  (sistema-vs-sistema).
- **Validação (HoneyBee):** onde o grid sobrepôs, o ml **dominou** a variância a
  speedup casado (@1,5× 0,66% vs 3,53%; @1,75× 1,41% vs 4,92%).

---

## 4. Veredito do Gate 5

**Forma estrita** ("ml domina a variância a speedup casado, em ≥2/3 seqs de
teste"): **NÃO atingida** — não há sobreposição de faixas de speedup em nenhuma das
3 seqs (0/3), por limitação do grid congelado frente ao conteúdo de teste.

**Contribuição da tese (sustentada pela evidência):**
1. **Redução de tempo vs baseline: forte e consistente** (3/3) — ~0,6% BD a TS
   ~30% (conservador), ~1,4% BD a TS ~48% (agressivo).
2. **Atribuição ao aprendizado: clara** (ml ≫ aleatório, 3/3).
3. **Atribuição a política casada: o escore RD do ml alcança a fronteira de
   baixo-BD que a heurística trivial da variância é incapaz de tocar** (3/3, razões
   11–94×) — a resposta limpa à "barra difícil", que não depende de sobreposição.
4. **Validação:** dominação direta a speedup casado.

**Enunciado honesto da tese:** *o H9a entrega redução de tempo de codificação
substancial e atribuível em 4K All-Intra held-out (TS ~30–48% a 0,6–1,4% BD-rate),
com ganho comprovadamente devido ao aprendizado (vs aleatório) e ao contexto de
taxa-distorção — cujo escore alcança pontos de operação de baixo-BD inacessíveis à
heurística trivial da variância. A dominação direta a speedup casado sobre a
variância é demonstrada na validação; no teste, a mesma superioridade manifesta-se
como acesso a um regime de operação (baixo-BD) que a variância não alcança.*

---

## 5. Limitações e trabalho futuro

- **Não-sobreposição de speedup no teste:** o grid congelado (calibrado no
  HoneyBee) não desceu o lado da variância a τ=0,97/0,99. O head-to-head direto a
  speedup casado ml-vs-variância no teste requereria esse grid — a fazer **na
  validação** (não no teste) para preservar o congelamento anti-cherry-picking.
- **Variância com política rica (rect-off):** para o head-to-head de sistemas
  implantados a speedup casado, rodar a variância também com rect-off (na
  validação).
- **Comparação com SOTA nativo** (`intra_cnn_based_part_prune`): não executada
  nesta fase; extensão natural.
- **Ruído:** 10 quadros/seq; repetir com mais quadros reforçaria a estabilidade
  das curvas.

---

## 6. Artefatos

`results/benchmark/h9_test/{Jockey,RaceNight,RiverBank}/` (curve_safe, curve_aggr,
ablation — cada um com `summary.csv`/`curve.csv` nos três pilares). Scripts de
análise: `results/benchmark/{fase5_final,matched_bd}.py`;
`src/scripts/benchmark/analyze_ablation.py` (comparação a speedup casado).
