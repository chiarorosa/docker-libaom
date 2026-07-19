# Resultados — H9c como substituto da CNN nativa (troca em cpu-used casado)

**Data:** 2026-07-19
**Estado:** resultado positivo, primeira agregação. Nenhum encode novo.
**Origem dos dados:** 192 encodes já em disco desde a campanha de troca, nunca agregados.

---

## 1. Por que este documento existe

Os 192 encodes de `results/benchmark/fase6_swap_h9c/raw_results.csv` (6 configurações × 8
sequências × 4 CQ) estavam medidos e sem nenhum agregado versionado. A causa é uma lista fixa de
configurações no gerador de relatórios: `report_swap.py:104` filtrava
`kind in ("h9a_bal", "h9a_aggr")` e descartava **em silêncio** toda linha H9c; o mesmo padrão existe
em `report_ctc.py:42` (`CONFIG_ORDER`). O filtro foi generalizado para descobrir as configurações a
partir do próprio CSV.

Este documento reporta o agregado, a decomposição por regime de CQ, os testes pareados por
sequência, a fronteira de Pareto sobre as 8 sequências e a divergência entre as duas definições de
TS em circulação nos documentos do projeto.

## 2. Procedência e método

| Item | Valor |
|---|---|
| Protocolo | AOM-CTC All-Intra, Classe A1, 4K, 10 bits |
| Sequências | 8 (BoxingPractice, Crosswalk, FoodMarket2, Neon1224, NocturneDance, PierSeaSide, Tango, TimeLapse) |
| Grade de CQ | 20, 32, 43, 55 |
| Âncora | libaom original, `cpu-used=0` — existe **apenas** em `fase6/raw_results.csv`; os lotes de troca não carregam âncora própria |
| BD-rate | Bjøntegaard sobre PSNR-Y, interpolação PCHIP (`src/scripts/benchmark/bd_rate.py`) |
| TS | média sobre CQ de `(t_âncora − t_cfg)/t_âncora`, depois média sobre sequências (definição canônica; ver §7) |

**Validade da comparação.** Dentro de um mesmo nível de `cpu-used`, `native_cpuN` e as trocas
compartilham todas as *speed features* exceto o podador de particionamento. As diferenças
intra-nível são, portanto, atribuíveis ao podador — é o que justifica o desenho de troca.

**Reprodução (contêiner):**

```bash
venv-ml/bin/python src/scripts/fase6/report_swap.py \
  --out-dir results/benchmark/fase6_swap_h9c
venv-ml/bin/python src/scripts/fase6/analyze_frontier.py \
  --out-dir results/benchmark/fase6_analysis
```

Saídas: `fase6_swap_h9c/{swap_per_seq,swap_average}.csv` + `swap_tables.tex`;
`fase6_analysis/{cq_decomposition,ts_per_cq,paired_tests,pareto_frontier,ts_definitions}.csv`.

---

## 3. Agregado sobre a grade CTC completa

| cpu-used | Podador | BD-rate (%) | TS (%) | Speedup |
|---|---|--:|--:|--:|
| 1 | CNN nativa | 0,449 | 32,59 | 1,508× |
| 1 | H9c τ=0,90 | 0,448 | 31,65 | 1,498× |
| 1 | H9c τ=0,95 | **0,414** | 30,34 | 1,469× |
| 2 | CNN nativa | 0,536 | 42,72 | 1,788× |
| 2 | H9c τ=0,90 | 0,539 | 42,07 | 1,787× |
| 2 | H9c τ=0,95 | **0,516** | 40,73 | 1,746× |
| 3 | CNN nativa | **2,722** | 67,94 | 3,159× |
| 3 | H9c τ=0,90 | 3,397 | 70,67 | 3,474× |
| 3 | H9c τ=0,95 | 3,384 | 70,25 | 3,419× |

Na média da grade completa, H9c e CNN nativa **empatam** em cpu1 e cpu2, e o H9c é claramente pior
em cpu3. O teste pareado de §5 sustenta essa leitura e **não** sustenta a afirmação mais forte de
que o H9c teria BD-rate menor.

---

## 4. Decomposição por regime de CQ

A média sobre a grade anula um efeito real e de sinal oposto entre as duas alavancas.

| Configuração | grade toda: BD / TS | cq20+32: BD / TS | cq43+55: BD / TS |
|---|--:|--:|--:|
| native_cpu1 | 0,449 / 32,59 | 0,153 / 25,26 | 0,630 / 39,93 |
| **h9c_tau95_cpu1** | 0,414 / 30,34 | **0,065 / 20,97** | 0,638 / 39,71 |
| h9c_tau90_cpu1 | 0,448 / 31,65 | 0,101 / 22,77 | 0,676 / 40,52 |
| native_cpu2 | 0,536 / 42,72 | 0,259 / 34,86 | 0,736 / 50,59 |
| **h9c_tau95_cpu2** | 0,516 / 40,73 | **0,173 / 30,98** | 0,764 / 50,49 |
| ml_balanced (cpu0) | 0,568 / 17,72 | 0,403 / 21,58 | 0,751 / 13,86 |
| ml_aggr (cpu0) | 1,403 / 31,51 | 1,068 / 38,33 | 1,774 / 24,69 |

No regime de alta qualidade o H9c τ=0,95 custa **menos da metade** do BD-rate da CNN nativa, em
ambos os níveis de `cpu-used` testados.

### 4.1 Mecanismo — dependência de CQ oposta

TS (%) por CQ individual, média das 8 sequências:

| Configuração | cq20 | cq32 | cq43 | cq55 |
|---|--:|--:|--:|--:|
| ml_balanced (cpu0) | **24,38** | 18,79 | 15,35 | 12,37 |
| ml_aggr (cpu0) | **41,82** | 34,84 | 27,93 | 21,46 |
| native_cpu1 | 22,44 | 28,07 | 41,22 | 38,64 |
| native_cpu2 | 32,53 | 37,19 | 50,27 | 50,90 |
| h9c_tau95_cpu1 | 16,84 | 25,10 | 40,27 | 39,15 |

As alavancas de ML em cpu0 têm TS **monotonicamente decrescente** em CQ; o degrau nativo é
**crescente**. As curvas cruzam entre cq20 e cq32: em cq20 o `ml_balanced` economiza mais tempo
(24,38%) que o degrau `cpu-used=1` inteiro (22,44%), e o `ml_aggr` economiza quase o dobro
(41,82% contra 22,44%). A ordem inverte a partir de cq32.

### 4.2 Explicação estrutural

Em All-Intra os três podadores nativos que agem **após** o candidato NONE estão desligados por
guarda de tipo de quadro:

| Podador | Local | Guarda |
|---|---|---|
| `av1_ml_predict_breakout` | `partition_search.c:4276` | `!frame_is_intra_only(cm)` |
| `av1_ml_early_term_after_split` | `partition_search.c:4338` | `!frame_is_intra_only(cm)` |
| `av1_ml_prune_rect_partition` | `partition_search.c:4351` | `!frame_is_intra_only(cm)` |

O nicho que o H9c ocupa — decidir depois de conhecer o custo RD real do NONE — é, em All-Intra,
**nativamente vazio**. Isso explica por que o H9c chega ao empate ou supera, enquanto o H9a compete
de frente com a CNN intra de `partition_strategy.c:207-280`, que nesse regime está ativa.

---

## 5. Testes pareados por sequência

Δ BD-rate (pontos percentuais) da troca contra `native_cpuN` no mesmo nível; negativo = troca
melhor. Pareado por sequência, n=8, teste t bilateral.

| Configuração | regime | Δ BD (pp) | se | t | p | melhor em |
|---|---|--:|--:|--:|--:|--:|
| h9c_tau95_cpu1 | grade toda | −0,034 | 0,029 | −1,18 | 0,278 | 4/8 |
| **h9c_tau95_cpu1** | **cq20+32** | **−0,088** | 0,036 | **−2,47** | **0,043** | **6/8** |
| h9c_tau95_cpu1 | cq43+55 | +0,009 | 0,038 | 0,23 | 0,826 | 3/8 |
| h9c_tau95_cpu2 | grade toda | −0,020 | 0,018 | −1,14 | 0,291 | 5/8 |
| **h9c_tau95_cpu2** | **cq20+32** | **−0,086** | 0,027 | **−3,19** | **0,015** | **7/8** |
| h9c_tau95_cpu2 | cq43+55 | +0,027 | 0,035 | 0,77 | 0,467 | 3/8 |
| h9c_tau90_cpu1 | cq20+32 | −0,052 | 0,033 | −1,58 | 0,157 | 6/8 |
| h9c_tau90_cpu2 | cq20+32 | −0,060 | 0,029 | −2,10 | 0,074 | 6/8 |
| h9c_tau95_cpu3 | grade toda | +0,662 | 0,160 | 4,14 | 0,004 | 0/8 |
| h9c_tau90_cpu3 | grade toda | +0,675 | 0,162 | 4,17 | 0,004 | 0/8 |

Para referência, todas as trocas H9a são significativamente **piores** que a nativa em todos os
regimes (Δ de +0,47 a +1,63 pp, p ≤ 0,01, 0/8 sequências melhores) — ver
`fase6_analysis/paired_tests.csv`.

**Leitura.** O efeito em alta qualidade **replica em dois níveis independentes** de `cpu-used`, e a
evidência mais forte está em cpu2 (p=0,015, melhor em 7 de 8 sequências), não em cpu1. Na grade
completa nada é significativo — é empate, não vitória. Em cpu3 a troca é significativamente pior.

---

## 6. Fronteira de Pareto sobre as 8 sequências

`SINTESE_resultados_metodologia.md §6` montava a fronteira sobre 3 sequências
(Boxing/FoodMarket2/Tango) embora as 8 já estivessem medidas. Recalculada sobre 17 configurações
completas em 8/8 sequências:

**Dominadas (4):** `ml_balanced`, `ml_aggr` — as duas configurações-carro-chefe em cpu0 —,
`h9a_bal_cpu1` e `h9c_tau90_cpu2`.

**Não dominadas (13):** entre elas `h9c_tau95_cpu1` (0,414% / 30,34%), que ocupa BD-rate **abaixo**
do primeiro degrau nativo (0,449%) com 93% do seu TS.

A "Conclusão 2" da síntese (o ML é dono do extremo de baixo BD) **se mantém e fica mais forte** ao
sair de 3 para 8 sequências. Mas o critério "não dominado na média" mostra-se fraco aqui: 13 de 17
configurações passam. Deve ser sempre combinado com o teste pareado de §5.

Tabela completa em `fase6_analysis/pareto_frontier.csv`.

---

## 7. As duas definições de TS em circulação

Os documentos do projeto usam **duas** definições de economia de tempo, sem sinalizar a diferença:

- **canônica** (`report_ctc.py`, `report_swap.py`, este documento):
  `TS = média_seq( média_CQ( 1 − t_cfg/t_âncora ) )`
- **da síntese** (`SINTESE_resultados_metodologia.md`):
  `TS = média_seq( 1 − Σ_CQ t_cfg / Σ_CQ t_âncora )`

| Configuração | TS canônico | TS da síntese | Δ (pp) |
|---|--:|--:|--:|
| ml_aggr | 31,51 | 34,14 | **+2,63** |
| ml_balanced | 17,72 | 19,26 | +1,53 |
| h9a_aggr_cpu1 | 51,82 | 52,13 | +0,30 |
| native_cpu1 | 32,59 | 30,42 | −2,17 |
| native_cpu2 | 42,72 | 40,37 | −2,36 |
| h9c_tau90_cpu1 | 31,65 | 29,05 | −2,60 |
| h9c_tau95_cpu1 | 30,34 | 27,53 | −2,81 |
| h9c_tau95_cpu2 | 40,73 | 37,76 | **−2,97** |

**O problema é maior que a magnitude.** A divergência não é um viés constante: vai de +2,63 pp a
−2,97 pp conforme a configuração, e portanto **reordena**. Sob a definição canônica,
`ml_aggr` (31,51) < `h9c_tau90_cpu1` (31,65) < `native_cpu1` (32,59); sob a da síntese,
`ml_aggr` (34,14) > `native_cpu1` (30,42) > `h9c_tau90_cpu1` (29,05) — o `ml_aggr` passa do mais
lento dos três ao mais rápido.

A causa é a mesma de §4: a definição da síntese soma tempos absolutos, o que pondera pelo CQ mais
caro (cq20), justamente onde o ML em cpu0 economiza mais. As duas definições são defensáveis
isoladamente, mas **não podem coexistir sem declaração** — a amplitude de ~3 pp é da mesma ordem
das lacunas que separam H9c da nativa.

**Recomendação:** adotar a canônica (é a dos CSVs e a do pipeline) e declarar a fórmula
explicitamente no capítulo de metodologia.

---

## 8. O que isto muda na tese

1. **A conclusão central deixa de ser negativa global e passa a condicional.** Não "o mecanismo
   nativo domina", mas: *o H9c é substituto drop-in da CNN nativa intra, com vantagem de qualidade
   estatisticamente significativa no regime de alta taxa (CQ 20/32), empate na grade completa e
   desvantagem em cpu-used=3.*
2. **A afirmação defensável sobre a grade completa é de paridade**, não de superioridade — e a
   paridade já é resultado: um MLP de 39 atributos sobre uma CNN convolucional sintonizada.
3. **A dependência de CQ é uma contribuição própria.** A literatura reporta médias sobre a grade
   CTC; o efeito documentado em §4 é cancelado por essa média e tem explicação estrutural
   verificada no código (§4.2), não apenas empírica.
4. **τ=0,90 contra τ=0,95 é ruído** e não deve figurar como dois pontos distintos de fronteira:
   o efeito médio é de −0,01 a −0,03 pp, com violações de sinal em várias sequências.

---

## 9. Limitações

- **Sem repetições.** Cada ponto é um encode único. O piso de ruído do tempo de parede foi
  *estimado* (σ ≈ 1–2% por encode) a partir de violações de monotonicidades que valem por
  construção, não medido. Diferenças de TS abaixo de ~2 pp não são resolvíveis em encode único; as
  de §3 sobrevivem porque a média pareada sobre 32 encodes tem erro-padrão ≈1,1%. Medir σ exige
  repetições (item E2 do plano de ações).
- **A vantagem de qualidade vem com custo de tempo.** Em cq20+32 o H9c τ=0,95 em cpu1 economiza
  20,97% contra 25,26% da nativa: troca-se ~4,3 pp de TS por ~0,09 pp de BD-rate. Se isso é bom
  negócio depende do ponto de operação desejado, e a fronteira de §6 é o instrumento correto para
  decidir — não a comparação isolada de BD.
- **8 sequências, uma classe.** Classe A1 4K 10 bits. Nada aqui se estende a outras resoluções sem
  medição.
- **O regime é All-Intra.** O mecanismo de §4.2 depende disso: em inter-frames os três podadores
  nativos pós-NONE voltam a atuar e o nicho do H9c deixa de estar vazio.
