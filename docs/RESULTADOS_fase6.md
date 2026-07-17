# Fase 6 — Resultados finais na CTC (validação universal) e extensão ML-vs-ML

Documento de resultados da tese. Cobre (i) a configuração experimental e a
**garantia de limpeza** dos resultados, (ii) os resultados finais na CTC
(libaom original vs H9a vs presets nativos) e (iii) a **extensão**: implantação
do H9a como substituto do pruner-CNN nativo do libaom nos perfis `cpu-used=1/2/3`.

Referências cruzadas: `docs/ANDAMENTO_tese.md` (§4.2 enquadramento), `docs/
RESULTADOS_fase5.md` (benchmark no universo do ML), `docs/PROTOCOLO_avaliacao.md`.

---

## 1. Configuração experimental (AOM-CTC, All Intra, Classe A1)

Seguindo o guia oficial `CWG-G082_AV2_CTC_v9` (§4.1 All Intra + §4 regra de
tiling 4K para A1), adaptado para o codificador **libaom** (o guia é do AVM/AV2;
a interface `--qp` do AVM não existe no libaom → usa-se `--cq-level`).

- **Conjunto de teste:** as 8 sequências da Classe A1 (vídeo natural, 4K,
  4:2:0, 10 bit): BoxingPractice, Crosswalk, FoodMarket2, Neon1224,
  NocturneDance, PierSeaSide, Tango, TimeLapse (fonte:
  `src/samples/aomctc_test_set/a1_4k_source.txt`; os `.y4m` ficam fora do git).
- **Frames:** 15 primeiros (`--limit=15`), todos intra.
- **Grade de QP:** `--cq-level ∈ {20, 32, 43, 55}` (grade da tese, mantida da
  Fase 5 para consistência com a validação; ver §nota-QP).
- **Modo:** good mode (`usage=0`) + `--kf-min-dist=0 --kf-max-dist=0`, fiel ao
  CTC (que **não** usa `--allintra`/`usage=2`).

### Comando de codificação (idêntico em todos os 192 encodes)

```
<enc>  --cpu-used=<C>  --passes=1  --end-usage=q  --cq-level=<Q> \
       --kf-min-dist=0  --kf-max-dist=0 \
       --deltaq-mode=0  --enable-tpl-model=0  --enable-keyframe-filtering=0 \
       --tile-columns=1  --tile-rows=0  --threads=2  --row-mt=0 \
       --bit-depth=10  --limit=15  --psnr  --obu \
       -o <out>.obu  <seq>.y4m
```

`--tile-columns=1` é log2 → **2 colunas** (regra 4K A1). Dimensões, fps e
bit-depth são lidos do header y4m. O PSNR-Y vem do próprio `--psnr` (sem passo
de `aomdec`). **Nota-QP:** `--cq-level` está na escala 0–63; a relação empírica
`qindex ≈ 4·cq` (verificada cq20→80) não bate exatamente nos qindex do CTC
(85/110/…); optou-se por manter a grade da tese (comparação com a Fase 5) em vez
dos qindex CTC, o que é registrado como escolha metodológica.

### As 6 configurações e os dois builds

| Config | Binário | `--cpu-used` | Env (taus do student) |
|---|---|:--:|---|
| `anchor` | `build/libaom_perf_anchor/aomenc` | 0 | — |
| `ml_balanced` | `build/libaom_perf/aomenc` | 0 | NONE=0.95 SPLIT=0.90 REST=0.20 |
| `ml_aggr` | `build/libaom_perf/aomenc` | 0 | NONE=0.60 SPLIT=0.85 REST=0.40 |
| `native_cpu1` | `build/libaom_perf_anchor/aomenc` | 1 | — |
| `native_cpu2` | `build/libaom_perf_anchor/aomenc` | 2 | — |
| `native_cpu3` | `build/libaom_perf_anchor/aomenc` | 3 | — |

- **`libaom_perf_anchor`** = `src/aom_baseline` (libaom v3.10.0 **puro**; o
  student **não** está compilado — `PARTITION_ML_STUDENT` ausente). Usado no
  anchor e nos três presets nativos.
- **`libaom_perf`** = `src/aom` com `-DPARTITION_ML_STUDENT=1` (student H9a
  embarcado, lê os taus por variável de ambiente). Usado nos dois pontos ML.

Script: `src/scripts/fase6/{run_fase6.sh, encode_ctc.py, report_ctc.py}`.
Saídas: `results/benchmark/fase6/{raw_results.csv, bdrate_per_seq.csv,
bdrate_average.csv, tables.tex, run.log}`.

---

## 2. Garantia de limpeza dos resultados

Ponto crítico de validade: os presets nativos `native_cpu1/2/3` **não** podem
ter o student H9a ativo (senão a comparação estaria contaminada). Verificado:

1. **O build usado nos native é o `libaom_perf_anchor`**, compilado de
   `src/aom_baseline` (v3.10.0 puro). O `CMAKE_HOME_DIRECTORY` do build aponta
   para `aom_baseline`; `PARTITION_ML_STUDENT` não está nas flags → o código do
   student **nem existe** nesse binário.
2. **Teste empírico byte-exato:** codificar 1 frame com e sem as variáveis
   `AV1_STUDENT_TAU_*` no `libaom_perf_anchor` produz saída **byte-idêntica**
   (mesmo tamanho, mesmo md5) → o student é ausente/inerte. Logo os native são
   o codificador nativo puro.
3. **A CNN de partição nativa (`intra_cnn_based_part_prune`) está LIGADA em
   `cpu-used=1, 2 e 3`.** Em `av1/encoder/speed_features.c`, o bloco
   `if (speed >= 1)` seta o nível para 2 e nenhum bloco de speed superior o
   zera (a lógica é cumulativa); em `cpu-used=0` (a âncora) ela está desligada
   (init = 0). Portanto os presets nativos representam o SOTA de poda-ML já
   embarcado, e a âncora é a busca completa sem poda-ML.

**Conclusão:** os pontos ML (cpu0, CNN nativa desligada, student ligado) e os
presets nativos (cpu≥1, CNN nativa ligada, student ausente) são configurações
limpas e disjuntas; a Fase 6 é metodologicamente válida.

---

## 3. Resultados finais na CTC (média sobre as 8 sequências A1)

Âncora = libaom original `cpu-used=0`. BD-Rate/BD-PSNR sobre PSNR-Y
(Bjøntegaard); TS% e speedup relativos à âncora.

| Config | BD-Rate | TS% | Speedup |
|---|--:|--:|--:|
| ML balanced (P_rect) | +0,57% | 17,7% | 1,22× |
| ML aggressive (A3) | +1,40% | 31,5% | 1,49× |
| libaom cpu-used=1 | +0,45% | 32,6% | 1,51× |
| libaom cpu-used=2 | +0,54% | 42,7% | 1,79× |
| libaom cpu-used=3 | +2,72% | 67,9% | 3,16× |

### Leitura honesta (fronteira taxa-BD × tempo)

Na CTC, o **botão de velocidade nativo domina os dois pontos ML** na fronteira:
`cpu-used=1` entrega mais economia de tempo (TS 32,6% vs 17,7% do balanced) por
BD-Rate igual ou menor; `cpu-used=1` também domina o ML aggressive (~mesmo TS a
1/3 do BD-Rate). Em 7/8 sequências o nativo fica à frente. A única nuance a favor
do ML é o **regime de baixíssimo speedup** (TS ~12-22%) que o passo discreto do
nativo (`cpu0→cpu1` já pula para ~33% TS) não cobre — p.ex. NocturneDance, ML
balanced +0,15% BD @ TS 12,6%, abaixo do nativo interpolado.

**Escopo do achado:** isto atinge a **utilidade prática** (capítulo de
Resultados), não a validação metodológica (Fases 2–5: o escore RD do modelo é
muito mais discriminativo que a variância sob política casada — mantém-se). Ver
§4 para a comparação de categoria correta.

---

## 4. Extensão — H9a como substituto do pruner-CNN nativo (cpu-used=1/2/3)

**Motivação.** Os presets `cpu-used=1/2/3` misturam a CNN-ML nativa com dezenas
de heurísticas não-ML; comparar o H9a contra eles responde a pergunta do
praticante ("por que não o botão nativo?"), mas **não** isola o mérito do
pruner. A comparação de categoria correta é **ML-vs-ML, tudo o mais constante**:
a `cpu-used=N` fixo, trocar *apenas* o pruner de partição — CNN nativa vs student
H9a — e medir o delta.

**Mecanismo (toggle `AV1_DISABLE_NATIVE_CNN`).** Adicionado em
`av1/encoder/partition_strategy.c` (sob `#if PARTITION_ML_STUDENT`): quando
`AV1_DISABLE_NATIVE_CNN` está setado, a chamada de `av1_intra_mode_cnn_partition`
é pulada (o student, que não tem gate de speed, assume como único pruner intra).
Leitura da env é cacheada; **default (sem set) é no-op byte-idêntico**.

**Verificação (1 frame, cpu-used=1, cq32, Tango; md5 do bitstream):**

| Config | md5 | Veredito |
|---|---|---|
| anchor (CNN on, sem student) | `187a3bc2416d` | baseline |
| perf, student inerte, sem toggle | `187a3bc2416d` | **== anchor → no-op safety** |
| perf, student inerte, CNN OFF | `3441fe5d4a62` | **≠ anchor → toggle desliga a CNN** |
| perf, balanced, CNN OFF (swap) | `699ce1ad400b` | **≠ demais → student assume** |

**Desenho do experimento.** A `cpu-used=N ∈ {1,2,3}`, três configurações que
compartilham *todas* as speed features exceto o pruner:

| Config | Binário | Env | Pruner | Fonte |
|---|---|---|---|---|
| `native_cpuN` | `libaom_perf_anchor` | — | CNN nativa (SOTA) | reaproveitado da Fase 6 |
| `h9a_bal_cpuN` | `libaom_perf` | `AV1_DISABLE_NATIVE_CNN=1` + balanced | student H9a | novo |
| `h9a_aggr_cpuN` | `libaom_perf` | `AV1_DISABLE_NATIVE_CNN=1` + aggressive | student H9a | novo |

O `native_cpuN` **não é re-executado**: seu bitstream é determinístico e o
ambiente (host + container) é **dedicado a esta tarefa desde o início**, mantendo
o wall-clock comparável entre execuções; assim reaproveita-se o `native_cpuN` e a
âncora `cpu-used=0` do CSV da Fase 6 (`results/benchmark/fase6/raw_results.csv`).
Só os 6 configs H9a rodam: **6 × 8 seqs × 4 cq × 15 frames = 192 encodes** (todos
cpu≥1, rápidos). O BD-Rate/TS referencia a mesma âncora `cpu-used=0`.

Scripts: `src/scripts/fase6/{run_swap.sh, encode_swap.py, report_swap.py}`.
Saídas: `results/benchmark/fase6_swap/{raw_results.csv, swap_per_seq.csv,
swap_average.csv, swap_tables.tex, run.log}`.

**Guardrail de honestidade.** Se a CNN nativa (isolada) dominar o H9a-swap, o
achado é "a poda-CNN embarcada é mais eficiente que o H9a" (caracteriza o SOTA e
o gap — ainda valioso). Se o H9a empatar/dominar em algum regime, é a vitória de
categoria correta que a tese busca. De qualquer forma é a comparação metodológica
apropriada, e complementa (não substitui) o resultado do praticante (§3).

### 4.1 Resultados do swap (média sobre as 8 sequências A1)

Âncora = `cpu-used=0` original (mesma referência do §3). BD-Rate/TS%/speedup por
`cpu-used` e configuração (`results/benchmark/fase6_swap/swap_average.csv`):

| cpu | config | BD-Rate | TS% | Speedup |
|:--:|---|--:|--:|--:|
| 1 | native (CNN nativa) | +0,45% | 32,59% | 1,51× |
| 1 | h9a_bal | +0,91% | 40,20% | 1,69× |
| 1 | h9a_aggr | +1,68% | 51,82% | 2,10× |
| 2 | native | +0,54% | 42,72% | 1,79× |
| 2 | h9a_bal | +1,03% | 50,05% | 2,05× |
| 2 | h9a_aggr | +1,80% | 60,97% | 2,61× |
| 3 | native | +2,72% | 67,94% | 3,16× |
| 3 | h9a_bal | +3,87% | 73,09% | 3,75× |
| 3 | h9a_aggr | +4,35% | 77,30% | 4,46× |

Padrão consistente no mesmo `cpu-used`: a CNN nativa isolada tem sempre BD-Rate
menor **e** TS/speedup menor que o H9a-swap — não há dominância direta no mesmo
nível, é um compromisso genuíno (o H9a corta mais tempo, ao custo de mais
BD-Rate).

### 4.2 Fronteira Pareto (os 9 pontos) e eficiência marginal

Minimizando BD-Rate e maximizando TS%, dos 9 pontos (3 níveis × 3
configurações), **apenas 1 é estritamente dominado**:

| # | config | BD-Rate | TS% | Pareto? | dominado por |
|--:|---|--:|--:|:--:|---|
| 1 | cpu1 native | 0,45% | 32,59% | sim | — |
| 2 | cpu2 native | 0,54% | 42,72% | sim | — |
| 3 | cpu1 h9a_bal | 0,91% | 40,20% | **não** | cpu2 native (BD-Rate menor **e** TS maior) |
| 4 | cpu2 h9a_bal | 1,03% | 50,05% | sim | — |
| 5 | cpu1 h9a_aggr | 1,68% | 51,82% | sim | — |
| 6 | cpu2 h9a_aggr | 1,80% | 60,97% | sim | — |
| 7 | cpu3 native | 2,72% | 67,94% | sim | — |
| 8 | cpu3 h9a_bal | 3,87% | 73,09% | sim | — |
| 9 | cpu3 h9a_aggr | 4,35% | 77,30% | sim | — |

Um único ponto do H9a (`cpu1 h9a_bal`) é cravado por `cpu2 native`. Os demais 8
pontos tecnicamente pertencem à fronteira — cada um oferece TS% maior que o
anterior a um BD-Rate maior — mas isso não significa que sejam bons pontos
operacionais: a métrica de retorno marginal (ΔTS / ΔBD entre pontos
consecutivos da fronteira, ordenados por BD-Rate crescente) revela onde o custo
compensa:

| trecho da fronteira | ΔBD-Rate | ΔTS% | ΔTS/ΔBD (marginal) |
|---|--:|--:|--:|
| cpu1 native → cpu2 native | 0,09 | 10,13 | **116,4** |
| cpu2 native → cpu2 h9a_bal | 0,49 | 7,33 | 14,8 |
| cpu2 h9a_bal → cpu1 h9a_aggr | 0,66 | 1,77 | **2,7** (pior trecho) |
| cpu1 h9a_aggr → cpu2 h9a_aggr | 0,12 | 9,15 | 76,9 |
| cpu2 h9a_aggr → cpu3 native | 0,92 | 6,97 | 7,6 |
| cpu3 native → cpu3 h9a_bal | 1,14 | 5,15 | 4,5 |
| cpu3 h9a_bal → cpu3 h9a_aggr | 0,48 | 4,21 | 8,7 |

Os dois trechos mais eficientes da fronteira inteira são puramente nativos
(cpu1→cpu2, 116,4) ou do H9a subindo de nível fixo (cpu1→cpu2 h9a_aggr, 76,9); o
trecho mais ineficiente é a transição entre configurações (`h9a_bal`→`h9a_aggr`
cruzando de cpu2 para cpu1), onde se paga bastante BD-Rate por pouco TS
adicional. Isto é, apesar de tecnicamente não-dominados, os pontos do H9a que
entram na fronteira têm retorno marginal pior que os saltos nativos — a
fronteira Pareto pura, sozinha, superestima o quão competitivo o H9a é.

### 4.3 Enquadramento do resultado para a tese

O H9a **não supera** a CNN nativa na fronteira taxa-BD × tempo, nem no cpu0
comparado aos presets (§3) nem isolando o pruner por `cpu-used` fixo (§4.1–4.2).
Isso não invalida a validação metodológica das Fases 2–5 (o escore RD do H9a
domina variância/aleatório sob política casada — mantém-se inalterado); é uma
resposta a uma pergunta mais dura: medir-se contra um produto industrial já
otimizado. Dois ângulos honestos permanecem defensáveis:

**(a) Custo de inferência (microbenchmark 2026-07-17).** Medição direta do custo
do **algoritmo de decisão isolado** de cada pruner (instrumentação
`AV1_PRUNER_TIMING`, encode real cpu1; `docs/RESULTADOS_microbench_pruner.md`): a
**inferência do MLP é ~50× mais barata por chamada** que a da CNN nativa
(~486 ns vs ~24.700 ns) — cerca de uma ordem e meia de grandeza. O H9a é uma MLP
por tamanho de bloco (36→64→32→3, ≈13,6 mil parâmetros no total); a CNN nativa é
convolucional multi-resolução (5 camadas + 4 ramos DNN). **Escopo:** compara-se a
inferência (passagem direta) dos modelos; a extração de features do MLP
(preprocessamento, otimizável/cacheável) e a frequência de invocação (por nó vs
1× por superbloco na CNN) são de integração, fora do escopo do algoritmo isolado.
O modelo aprendido leve executa sua predição a fração do custo da CNN de
produção — este é o ângulo de custo defensável, complementar à granularidade fina
(§3) e à paridade de qualidade do H9c com a CNN nativa a cpu1/2.

**(b) Nicho de baixo speedup.** Como já registrado em §3, o degrau discreto dos
presets nativos (`cpu0→cpu1` já salta para TS~33%) deixa uma lacuna em regime de
baixíssimo speedup (TS ~12–22%) que o H9a cobre continuamente via calibração de
τ — p.ex. NocturneDance, ML balanced +0,15% BD @ TS 12,6%, abaixo do que o
nativo interpolado ofereceria nesse ponto. O H9a não precisa vencer a fronteira
nativa inteira para ser útil: preenche uma granularidade de operação que o
degrau grosseiro dos presets não alcança.

**Síntese:** o H9a não desloca o botão de velocidade nativo como produto
final, mas (i) entrega parte relevante do ganho de tempo a uma fração do custo
computacional de uma CNN de produção, e (ii) cobre um regime operacional de
baixo speedup que os presets nativos pulam. É a caracterização honesta do gap
frente ao SOTA embarcado, mantendo intacta a validação metodológica (Fases
2–5) de que o contexto RD é um sinal muito mais discriminativo que heurísticas
triviais.
