# Inventário consolidado — todas as soluções propostas, todas as configurações testadas

**Data:** 2026-07-26
**Propósito.** Tabela agregada por família de solução, com **cada variação de configuração
testada**, resultados (BD-rate, TS, speedup) quando cabível, e o estado de cada uma:
integrada em C? em que portão parou? medida em quê?

---

## 0. Como ler estas tabelas

**Âncora.** Todo BD-rate / TS / speedup é contra **libaom v3.10.0 puro, `cpu-used=0`,
All-Intra** (`libaom_perf_anchor`), salvo indicação em contrário na própria linha.

**Definição de TS — atenção, há duas na tese.** `analyze_frontier.py:281` computa ambas:

- **canônica:** média sobre CQ de `(1 − t_cfg/t_âncora)`, depois média sobre sequências;
- **"sintese":** média sobre sequências de `(1 − Σ_CQ t_cfg / Σ_CQ t_âncora)` — ponderada
  pelo tempo, logo dominada por CQ 20.

Divergem até **~3 pp** (`results/benchmark/fase6_analysis/ts_definitions.csv`). **Todas as
tabelas deste documento usam a canônica**, que é a adotada pelos documentos mais recentes
(`RESULTADOS_H9d_CTC.md`, `RESULTADOS_BLOCO7_E1_E4.md`). A tabela de `SINTESE §4` usa a
outra — por isso o preset nativo aparece lá como 30,42% / 1,440× e aqui como 32,59% /
1,508×. **Não é discrepância de medição, é de definição.**

**Escada de portões** (`PROTOCOLO_avaliacao.md §6`, congelado):

| portão | o que testa |
|---|---|
| **Gate 2** | sinal offline na simulação oráculo, risco casado, validação |
| **Gate 3** | validação held-out (oráculo, SPLIT-lost casado) |
| **Gate 4 / C** | integração em C: paridade bit-a-bit C↔Python, no-op byte-idêntico |
| **Gate 5** | benchmark real no teste held-out (BD × tempo) |
| **Fase 6** | validação universal na CTC (AOM CTC v9, Classe A1) |

**Partições de dados (congeladas, sem vazamento).** UVG 4K: treino 10 seqs · **validação**
HoneyBee, FlowerPan, Lips · **teste** Jockey, RaceNight, RiverBank. **CTC**: 8 seqs Classe
A1 4K 10-bit (BoxingPractice, Crosswalk, FoodMarket2, Neon1224, NocturneDance, PierSeaSide,
Tango, TimeLapse), 15 quadros.

**Coluna "Impl. C"** = existe no fork da libaom sob `#if PARTITION_ML_STUDENT` (ou env-gate
equivalente), com no-op byte-idêntico garantido quando desligado.

---

## 1. Referências e linhas de base (não são propostas — são a régua)

| Config | BD-rate | TS% | Speedup | Impl. C | Portão | UVG val/test | Enc. UVG | Enc. CTC |
|---|--:|--:|--:|:--:|---|---|:--:|:--:|
| **Âncora** libaom cpu0 puro | 0 | 0 | 1,000× | n/a (é a âncora) | — | ambos | 3 (teste) | 8 |
| libaom `cpu-used=1` | +0,449% | 32,59 | 1,508× | nativo | — | — | — | 8 |
| libaom `cpu-used=2` | +0,536% | 42,72 | 1,788× | nativo | — | — | — | 8 |
| libaom `cpu-used=3` | +2,722% | 67,94 | 3,159× | nativo | — | — | — | 8 |
| CNN nativa intra (`intra_cnn_based_part_prune`) | — (medida via presets) | — | — | nativo | SOTA embarcado | — | 3 | 8 |
| Heurística de variância `exp(−var/1000)` | ver §2 | — | — | sim (braço de ablação) | piso de pixels | val+teste | 1 (2 quadros) | — |
| Aleatório | — | — | — | sim (braço de ablação) | piso absoluto | teste | 1 (2 quadros) | — |

> A CNN nativa não tem linha própria de BD/TS: ela é ligada por *speed feature* dentro dos
> presets, e só é isolável desligando-a (`AV1_DISABLE_NATIVE_CNN=1`), o que se faz nas
> tabelas de *swap* (§3.3, §4.3).

---

## 2. Família A — domínio de pixels (luma como entrada)

**Estado da família: FECHADA, cinco tentativas independentes negativas.** Nenhuma via de
pixels aprendida de ponta a ponta foi implantada.

| Config | BD-rate | TS% | Speedup | Impl. C | Portão | UVG val/test | Enc. UVG | Enc. CTC |
|---|--:|--:|--:|:--:|---|---|:--:|:--:|
| **ConvNeXt substituto (CE)** — replay H8, τ 0,90/0,90/0,20 | −0,11% | ~2,4–3,3 | 1,02× | **não** (replay pelo gancho, sem inferência conv. em C) | mediu teto; **não é cota superior** (perde p/ `pixels24`) | teste | 1 (Jockey) | — |
| ConvNeXt substituto — sweep H8 agressivo | até +1,62% | — | até 1,29× | não | idem | teste | 1 | — |
| **ConvNeXt com alvo de *regret*** (α=3) | — | — | — | não | **refutado no crivo A5** (pior que CE em toda a faixa, 1,06–3,80×) | val+teste (6 seqs) | 0 | — |
| **`pixels24`** (estudante MLP de 24 atributos, era H7) | ver §2.1 | — | — | sim (era H7, substituído pelo H9a) | Gate 2: 10–19% de redução de custo | val+teste | 3 | — |
| **GNN / Approach B** — não-causal | — | — | — | não | fura o oráculo (+28 pp) mas **não é causal** | val+teste | 0 | — |
| **GNN / Approach B** — *deployable* pixel-only | ~1,5% @ TS casado | — | — | não (replay H8) | **Gate 5 refuta**: H9a domina ~2× em BD em todo o sweep de τ | teste | 1 (Jockey, 5 qd) | — |
| **Bloco D** — SATD do bloco-**fonte** ("H9b") | — | — | — | não | **Gate 2 nulo** (46,9/50,5/53,4/57,8 vs H9a 47,0/49,7/52,6/57,3) — e **não era o atributo especificado** | val | 0 | — |
| **Bloco D'** — predizibilidade intra a partir dos **vizinhos** | — | — | — | não | **portão não passa** (26/07): nulo em 16 px com 15 855 nós | val | 0 | — |

### 2.1 Ablação de atribuição (a escada `aleatório → variância → pixels24`)

Política de poda idêntica, variando **só a fonte do escore**. Único experimento de
codificação da família; **2 quadros, 1 sequência** — a limitação que motivou o item E5.

| Fonte do escore | BD-rate @ speedup 1,3× | Impl. C | Enc. UVG |
|---|--:|:--:|:--:|
| variância `exp(−var/1000)` | **0,76%** | sim | 1 seq, 2 quadros |
| `pixels24` ("ML" da época) | 1,39% | sim | 1 seq, 2 quadros |

> **A variância vence o modelo de pixels aqui** — o resultado negativo que motivou a virada
> H9. **Ressalva registrada:** o crivo A5 (6 seqs, 792 mil nós) **contradiz**, com
> `pixels24` batendo a variância. A contradição está declarada e não resolvida; resolvê-la
> é o item **E5** (pausado).

### 2.2 Hierarquia no crivo A5 (métrica de *regret*, offline, 6 seqs held-out)

`reg_frac` em `cost_red` casado de 25% — **menor é melhor**:

| modelo | `reg_frac` | leitura |
|---|--:|---|
| `variance` | 0,0573 | piso |
| `convnext_regret` | 0,0219 | pior que a CE |
| `convnext_ce` | 0,0207 | 28,1 M parâmetros sobre pixels crus |
| `pixels24` | 0,0121 | **melhor modelo de pixels medido** |
| `H9a` | 0,0036 | o implantado |

> **Nota de composição (auditoria de 26/07).** `pixels24` **é o bloco A do próprio H9a**;
> 24 dos 36 atributos do H9a são descritores de luma. A leitura correta não é "contexto RD
> vence pixels", e sim: descritores manuais compactos vencem a CNN profunda, e os 12
> atributos de vizinhança/quantização/posição agregam 3,4× **sobre** eles. Ver
> `RESULTADOS_auditoria_dominio_pixels.md`.

---

## 3. Família B — H9a: poda pré-busca (a solução IMPLANTADA)

**Estado: IMPLANTADA.** 36 atributos (A+B+C), MLP por nível 36→64→32→3, CE de rótulo duro.
Portões 2, 3, 4 e 5 **todos passaram**. Ação em cascata por τ: NONE-commit → split-quadrada →
rect-off.

### 3.1 Configurações de τ — teste held-out UVG (Fase 5, 3 seqs, 10 quadros, cpu0)

| Config (τ none/split/rest) | BD-rate | TS% | Speedup | Impl. C | Portão | UVG | Enc. UVG | Enc. CTC |
|---|--:|--:|--:|:--:|---|---|:--:|:--:|
| **P0** conservador (0,84/0,90/−) | 0,155% | 19,05 | ~1,24× | sim | Gate 5 ✅ | teste | 3 | — |
| **P_rect** equilibrado (0,95/0,90/0,20) | 0,464% | 26,49 | ~1,36× | sim | Gate 5 ✅ | teste | 3 | — |
| **P_ref** refinado por nível | 0,595% | 29,53 | ~1,42× | sim | Gate 5 ✅ | teste | 3 | — |

### 3.2 Configurações de τ — CTC (Fase 6, 8 seqs, 15 quadros, cpu0)

| Config | BD-rate | TS% | Speedup | Impl. C | Portão | Enc. CTC |
|---|--:|--:|--:|:--:|---|:--:|
| **`ml_balanced`** (P_rect) | +0,568% | 17,72 | 1,223× | sim | Fase 6 ✅ | 8 |
| **`ml_aggr`** (A3) | +1,403% | 31,51 | 1,492× | sim | Fase 6 ✅ | 8 |

### 3.3 *Swap* — H9a substitui a CNN nativa (`AV1_DISABLE_NATIVE_CNN=1`), CTC 8 seqs

| cpu | Config | BD-rate | TS% | Speedup | Enc. CTC |
|:--:|---|--:|--:|--:|:--:|
| 1 | H9a balanced | 0,915% | 40,20 | 1,694× | 8 |
| 1 | H9a aggressive | 1,685% | 51,82 | 2,104× | 8 |
| 2 | H9a balanced | 1,030% | 50,05 | 2,046× | 8 |
| 2 | H9a aggressive | 1,805% | 60,97 | 2,610× | 8 |
| 3 | H9a balanced | 3,866% | 73,09 | 3,754× | 8 |
| 3 | H9a aggressive | 4,347% | 77,30 | 4,465× | 8 |

> **Leitura:** o H9a-swap poda mais que a nativa mas a **BD desproporcional (~2–3×)** →
> a CNN nativa é mais eficiente como podador.

### 3.4 Pontos históricos do H7 (Jockey, 5 quadros — antes da virada H9)

| Config | BD-rate | Speedup | Enc. UVG |
|---|--:|--:|:--:|
| P0 (só NONE-commit) | 0,25% | 1,03× | 1 |
| P_rect (+rect-off) | 0,49% | 1,05× | 1 |
| P_ref (por nível) | 0,42% | 1,07× | 1 |
| A1 (0,80/0,90/0,20) | 0,42% | 1,08× | 1 |
| A2 (0,70/0,90/0,30) | 1,23% | 1,18× | 1 |
| A3 (0,60/0,85/0,40) | 1,62% | 1,29× | 1 |

### 3.5 Ablações de modelagem SOBRE o H9a (todas offline; nenhuma implantada)

| Config | Resultado | Impl. C | Portão | UVG | Enc. |
|---|---|:--:|---|---|:--:|
| **B1** — contexto RD hereditário (pai + irmãos, 42 atributos) | **negativo** — degrada a decisão ponderada por custo em toda a fronteira | não | crivo A5 | val+teste | 0 |
| **B2** — τ adaptativo por qindex | **positivo mas imaterial** — ganho absoluto minúsculo | não | crivo offline (48 mil superblocos) | val+teste | 0 |
| **B3** — direcional HORZ vs VERT | **negativo** — acurácia direcional 69,2%→69,5% pareado; recall ~42%→~47% | não | portão de modelagem | val+teste | 0 |
| **B4** — ponderação de classe por nível | **negativo** — troca ganho por risco sem melhorar o desfecho | não | crivo A5 | val+teste | 0 |

### 3.6 Diagnósticos de engenharia (Bloco 6, série C)

| Item | Pergunta | Resultado | Impl. C | Enc. UVG |
|---|---|---|:--:|:--:|
| **C1/C3** | AB/4-way têm custo a recuperar? | **sim** — 34,3% agregado (mín. 28,9%), 3–4× acima do limiar de 10% → justifica o H9d | diagnóstico | 3 (teste) |
| **C2** | de onde vem o ganho, por nível? | decomposição 16/32/64 por env-var | sim (env) | 3 |
| **C5** | poda *soft* melhoraria a fronteira? | **falsificada** — a fronteira de τ já é densa (gaps ≤0,15×); cirurgia em C não se justifica | não (falsificada antes) | 3 |
| **A4** | calibração da softmax implantada | 1 816 393 nós de decisão, teste congelado | n/a | 0 |

---

## 4. Família C — H9c: refinamento pós-NONE (contexto RD **real**)

**Estado: IMPLANTADO em C, mas NÃO é contribuição autônoma.** 39 atributos (A+B+C+E), ação
binária. Gate 3 passou com folga (61,2% de redução de custo a 0,20% de SPLIT-lost); Gate C
(paridade + no-op) passou.

### 4.1 CTC a cpu0 (8 seqs) — o extremo de baixo BD

| Config | BD-rate | TS% | Speedup | Impl. C | Portão | Enc. CTC |
|---|--:|--:|--:|:--:|---|:--:|
| **`h9c_tau95`** | **+0,160%** | 12,6 | 1,15× | sim | Fase 6 ✅ | 8 |
| **`h9c_tau90`** | **+0,171%** | 13,6 | 1,16× | sim | Fase 6 ✅ | 8 |
| `h9c_tau45` (sonda do joelho, E3) | dominado por `native_cpu1` | — | — | sim | **negativo** — τ∈(30,60) não contém nada | subconjunto CQ20 |

### 4.2 O *confound* do H9a — quanto do "TS do H9c" era o H9a (Neon1224, cpu0)

| Config | BD-rate | TS% |
|---|--:|--:|
| H9a@0,9 sozinho | 0,312% | 17,10 |
| H9a@0,9 + H9c | 0,270% | 17,36 |
| **H9c sozinho** | 0,037% | **4,23** |

**E4, ampliado a 4 seqs CTC:** em média **64% do TS atribuído ao H9c era o H9a@default**
(75% / 58% / 28% / 95% em Neon1224 / PierSeaSide / Tango / TimeLapse). Empilhar H9c sobre
H9a adiciona **+0,26 pp** de TS.

### 4.3 *Swap* limpo — H9c substitui a CNN nativa (H9a neutralizado), CTC 8 seqs

| cpu | Config | BD-rate | TS% | Speedup | Enc. CTC |
|:--:|---|--:|--:|--:|:--:|
| 1 | H9c τ0,95 | 0,414% | 30,34 | 1,469× | 8 |
| 1 | H9c τ0,90 | 0,448% | 31,65 | 1,498× | 8 |
| 2 | H9c τ0,95 | 0,516% | 40,73 | 1,746× | 8 |
| 2 | H9c τ0,90 | 0,539% | 42,07 | 1,787× | 8 |
| 3 | H9c τ0,95 | 3,384% | 70,25 | 3,419× | 8 |
| 3 | H9c τ0,90 | 3,397% | 70,67 | 3,474× | 8 |

> **Leitura:** como substituto da CNN nativa, o H9c **empata com ela a cpu1 e cpu2** — a
> τ0,95 tem BD *menor* que a nativa com um pouco menos de TS. É o cenário em que tem mérito
> próprio. A cpu3 perde.

---

## 5. Família D — H9d: podador pós-NONE de partições ESTENDIDAS (AB / 4-way)

**Estado: IMPLANTADO, Pareto-não-dominado, contribuição marginal positiva e barata.**
Alvo distinto das demais: não decide NONE/SPLIT, decide se vale avaliar AB={4,5,6,7} e
4-way={8,9}.

### 5.1 Portão de predizibilidade (offline, 792 840 nós)

| nível | n | base EXT | **ROC-AUC** | PR-AUC |
|---|--:|--:|--:|--:|
| 16 px | 572 213 | 9,9% | **0,906** | 0,452 |
| 32 px | 172 627 | 13,1% | 0,817 | 0,359 |
| 64 px | 48 000 | 3,0% | 0,864 | 0,155 |
| **agregado** | **792 840** | 10,2% | **0,890** | 0,425 |

Com 39 atributos (contexto RD pós-NONE) o AUC sobe para **0,902**. Veredito: **GO**.

### 5.2 Cota superior — desligar AB/4-way por completo (`AV1_EXT_PART_OFF=1`, 3 seqs UVG teste, 5 quadros)

| sequência | BD-rate | Speedup (teto) |
|---|--:|--:|
| Jockey | +1,13% | 1,485× |
| RaceNight | +0,76% | 1,418× |
| RiverBank | +0,79% | 1,402× |
| **média** | **+0,89%** | **1,431×** |

### 5.3 Sweep de τ no codificador (UVG teste, 5 quadros, vs âncora cpu0)

| Config | Jockey | RaceNight | RiverBank | Impl. C | Enc. UVG |
|---|--:|--:|--:|:--:|:--:|
| H9a P_ref (base) | 1,41% / 1,46× | 0,32% / 1,50× | 0,16% / 1,29× | sim | 3 |
| H9a+extoff (*blanket*) | 2,19% / 2,00× | 1,21% / 1,86× | 0,91% / 1,66× | sim | 3 |
| **H9a+H9d τ=0,10** | 1,48% / 1,49× | 0,55% / 1,61× | 0,19% / 1,32× | sim | 3 |
| **H9a+H9d τ=0,20** | 1,70% / 1,61× | 0,70% / 1,69× | 0,30% / 1,38× | sim | 3 |
| **H9a+H9d τ=0,30** | 1,72% / 1,77× | 0,90% / 1,75× | 0,44% / 1,45× | sim | 3 |
| **H9a+H9d τ=0,45** | 2,02% / 1,87× | 1,08% / 1,80× | 0,64% / 1,55× | sim | 3 |

### 5.4 τ por nível (refinamento que recupera o RiverBank), 10 quadros

| Config | Δ BD Jockey | Δ BD RaceNight | Δ BD RiverBank |
|---|--:|--:|--:|
| τ global 0,30 | −0,29 | −0,34 | **+0,12 (pior)** |
| **PL10** (θ por nível @wl~10%) | −0,02 | **−0,30** | **+0,05 (~empate)** |
| PL20 (@wl~20%) | +0,03 | −0,12 | +0,04 |
| PLmix (agressivo-16) | −0,00 | −0,08 | +0,03 |

**Escolhido: PL10** (τ_16=0,091 · τ_32=0,103 · τ_64=0,014).

### 5.5 CTC (8 seqs, 15 quadros) — o resultado que entra nos Resultados

| Config | BD-rate | TS% | Speedup | Impl. C | Portão | Enc. CTC |
|---|--:|--:|--:|:--:|---|:--:|
| H9a P_rect (base) | +0,568% | 17,72 | 1,223× | sim | Fase 6 ✅ | 8 |
| **H9a + H9d (PL10)** | **+0,586%** | **18,74** | **1,238×** | sim | **Fase 6 ✅** | 8 |

**Contribuição marginal:** +1,02 pp de TS por +0,018 pp de BD = **0,018 pp/pp**, contra
**0,063 pp/pp** do knob de τ do H9a — ou seja, **~3,5× mais barato** que subir τ. Vence em
**6 de 8** sequências, 2 por Pareto estrito. Integridade verificada: com H9d desligado, o
bitstream reproduz `ml_balanced` **byte-idêntico** (1 574 775 bytes, PSNR-Y 40,9720 dB).

---

## 6. Família E — reformulações do problema (todas negativas)

| Config | Resultado | Impl. C | Portão | UVG | Enc. |
|---|---|:--:|---|---|:--:|
| **Solução 4** — regressão de *regret* (prever custo em vez de classificar) | **refutada** — não alcança a faixa de risco baixo: 0,00% de redução de custo @0,5% SPLIT-lost, contra **51,73%** do classificador com as **mesmas** features | não | **Gate 3 reprova**; Gate 5 não foi pago | val | 0 |
| **Solução 4** — variante *naïve* | 0,00% @0,5%; menor SPLIT-lost 12,4% (degenerado por zero-inflação) | não | Gate 3 | val | 0 |
| **Solução 4** — variante balanceada (Huber ponderado, peso 8–46×) | 0,00% @0,5%; menor SPLIT-lost 11,4% | não | Gate 3 | val | 0 |

> **Valor metodológico:** dá base empírica a *por que* a formulação de **classificação**
> (H9a) é a correta — ~11% dos nós *true-SPLIT* têm *regret* ≈ 0, indistinguíveis dos
> seguros.

---

## 7. Resumo — o que sobreviveu

| Família | Configs testadas | Implantado? | Estado final |
|---|:--:|:--:|---|
| **A — pixels** (ConvNeXt CE/regret, pixels24, GNN, bloco D, bloco D') | 8 | **não** | **fechada** — 5 tentativas independentes negativas |
| **B — H9a** (P0/P_rect/P_ref/A1–A3, ml_balanced/aggr, 6 swaps) | 15 + 4 ablações + 4 diagnósticos | **sim** | **a solução da tese**; portões 2–5 e Fase 6 |
| **C — H9c** (τ0,45/0,90/0,95, iso, 6 swaps) | 10 | **sim** | positivo **como substituto da CNN nativa** a cpu1/2; **não** é contribuição autônoma (64% do TS era o H9a) |
| **D — H9d** (blanket, 4 τ globais, 4 τ por nível, CTC) | 10 | **sim** | **2ª solução positiva**; +1,02 pp TS por +0,018 pp BD, 3,5× mais barato que o knob de τ |
| **E — reformulações** (regressão de *regret*, 3 variantes) | 3 | não | refutada no Gate 3 |

**Duas soluções positivas implantadas: H9a (principal) e H9d (complemento).** O H9c é
implantado e mede-se bem contra a CNN nativa, mas não sobrevive como contribuição
independente.

---

## 8. Lacunas conhecidas neste inventário

- **Ablação de atribuição (§2.1) é de 2 quadros / 1 sequência** e é contradita pelo crivo
  A5. Resolvê-la é o item **E5**, hoje **pausado por decisão**.
- **Fronteira do H9d na CTC** tem um único ponto (PL10 × P_rect); as 3 configs restantes
  (PL20 × P_rect, PL10 × A3, PL20 × A3, ~9 h) são **confirmatórias** e não foram rodadas.
- **A CNN nativa não tem linha isolada de BD/TS** — só é observável por diferença nos
  *swaps*.
- **Custo de inferência do ConvNeXt nunca foi pago** (`surrogate_replay.py:8`: *no
  convolutional inference in C*): as linhas H8 da §2 são limites superiores que ignoram
  custo.
- **Reprodutibilidade temporal:** σ do TS medida em `RESULTADOS_BLOCO7_E3_DEC_E2.md`
  (CV 0,28%, resolução ~0,46 pp) — deltas de TS abaixo disso não são resolvíveis. O BD-rate
  é exato (bytes e PSNR determinísticos).
