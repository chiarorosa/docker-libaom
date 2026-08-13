# Tabela completa dos atributos da família H9

> Materialização, atributo a atributo, do vetor que alimenta os podadores da
> família H9. Os nomes e as descrições estão **em inglês**, na forma exata em que
> aparecem no código que os produz. Quando a grandeza é lida de estado do
> **libaom**, a coluna de origem traz o símbolo exato (arquivo e linha) e a coluna
> de descrição reproduz **literalmente** o comentário do libaom, entre aspas,
> antes de qualquer texto derivado (que vem após "→").
>
> Nenhuma descrição foi inventada. Onde o libaom não documenta o campo, isso está
> declarado na própria célula.

**Procedência.** Fonte única de verdade dos atributos:
`src/scripts/partition_model/features.py` (docstring do módulo, linhas 23–46;
`FEATURE_NAMES` 53–61; `H9_FEATURE_NAMES` 205–211; `H9_SUBSETS` 214–220;
`NUM_FEATURES_H9A = 36` linha 204; `NUM_FEATURES_H9C = 39` linha 382). Espelho em
C, com paridade verificada: `src/aom/av1/encoder/partition_strategy.c`,
`student_feats18` (1813–1885), `student_node_features` (1893–1992),
`student_h9c_decide` (2192–2238) e `student_h9d_decide` (2244–2266). Símbolos do
libaom citados em `av1/common/blockd.h`, `av1/common/common_data.h`,
`av1/common/av1_common_int.h`, `av1/common/quant_common.{h,c}`,
`av1/encoder/block.h` e `av1/encoder/encodeframe_utils.h` (tag v3.10.0).
Composição dos conjuntos descrita em `M4_atributos_e_politica.md` §4.1 e §4.8.

---

## 1. Como ler a coluna "Incluso em"

| Solução | Ponto de enganche no codificador | Vetor | Nº de atributos | Construtor (Python) | Espelho (C) |
|---|---|---|--:|---|---|
| **H9a** | `av1_prune_partitions_before_search` (pré-busca) | A + B + C | **36** | `node_features_h9a` | `student_node_features` |
| **H9c** | `av1_prune_after_none` (pós-`PARTITION_NONE`) | A + B + C + E | **39** | `node_features_h9c` | `student_h9c_decide` |
| **H9d** | `av1_prune_after_none` (pós-`PARTITION_NONE`) | A + B + C + E | **39** | `node_features_h9c` | `student_h9d_decide` |

**H9c e H9d consomem exatamente o mesmo vetor de 39 atributos**, no mesmo ponto de
enganche; diferem apenas na ação (encerrar a busca *versus* pular as partições
estendidas AB/4-way) e nos pesos treinados. Por isso as duas soluções aparecem
sempre juntas na coluna "Incluso em": não existe atributo que esteja em uma e não
na outra. Registre-se ainda que os rótulos H9a/H9b/H9c nomeiam **conjuntos de
atributos**, ao passo que H9d nomeia uma **ação** — dois esquemas de nomeação
convivendo na mesma família (`M4` §4.8).

**Blocos.** A = pixels de luminância do nó, do pai e dos irmãos (+ `q_norm` e
posição no superbloco); B = vizinhança de particionamento causal; C =
quantização e posição no quadro; D = proxy de SATD (**nunca implantado**, ver §3);
E = contexto de taxa-distorção real do `PARTITION_NONE`.

**Convenções da tabela.** `n` é o lado do bloco em pixels
(`block_size_wide[blk->bsize]`); `var` é a variância populacional calculada por
somas inteiras exatas; todo `log1p` é o da libc. O índice da coluna `Idx` é o do
**vetor implantado**; no vetor completo de 41 atributos usado apenas na ablação
offline do Gate 2, o bloco E ocupa os índices 38–40, porque o bloco D ocupa 36–37.

---

## 2. Tabela 1 — Atributos implantados (blocos A, B, C e E)

| Idx | Bloco | Feature name (código) | Origem no libaom (símbolo exato) | Description (English) | Incluso em |
|--:|:--:|---|---|---|---|
| 0 | A | `log_var` | `x->plane[AOM_PLANE_Y].src.buf` — "A buffer containing the source frame." (`av1/encoder/block.h:143`) | `log1p(var)` — "overall texture / flatness" | H9a, H9c, H9d |
| 1 | A | `log_var_q0` | idem (quadrante superior esquerdo do bloco) | `log1p(var)` of quadrant 0 — "per-quadrant texture" | H9a, H9c, H9d |
| 2 | A | `log_var_q1` | idem (quadrante superior direito) | `log1p(var)` of quadrant 1 — "per-quadrant texture" | H9a, H9c, H9d |
| 3 | A | `log_var_q2` | idem (quadrante inferior esquerdo) | `log1p(var)` of quadrant 2 — "per-quadrant texture" | H9a, H9c, H9d |
| 4 | A | `log_var_q3` | idem (quadrante inferior direito) | `log1p(var)` of quadrant 3 — "per-quadrant texture" | H9a, H9c, H9d |
| 5 | A | `quad_var_spread` | idem | `(max_qv - min_qv)/(max_qv+1)` — "quadrant-variance spread" | H9a, H9c, H9d |
| 6 | A | `log_var_qsums` | idem | `log1p(var of quadrant sums)` — "brightness heterogeneity across quadrants" | H9a, H9c, H9d |
| 7 | A | `log_var_qvars` | idem | `log1p(var of quadrant vars)` — "texture heterogeneity across quadrants" | H9a, H9c, H9d |
| 8 | A | `log_hgrad` | idem | `log1p(horizontal grad sum)` — "vertical-edge energy" | H9a, H9c, H9d |
| 9 | A | `log_vgrad` | idem | `log1p(vertical grad sum)` — "horizontal-edge energy" | H9a, H9c, H9d |
| 10 | A | `grad_orient` | idem | `(hgrad - vgrad)/(hgrad+vgrad+1)` — "gradient orientation in [-1,1]" | H9a, H9c, H9d |
| 11 | A | `log_var_rowsums` | idem | `log1p(var of row sums)` — "horizontal-band structure (HORZ cue)" | H9a, H9c, H9d |
| 12 | A | `log_var_colsums` | idem | `log1p(var of col sums)` — "vertical-band structure (VERT cue)" | H9a, H9c, H9d |
| 13 | A | `rowcol_orient` | idem | `(vrow - vcol)/(vrow+vcol+1)` — "row-vs-col structure orientation" | H9a, H9c, H9d |
| 14 | A | `log_maxgrad` | idem | `log1p(max \|grad\|)` — "strongest edge" | H9a, H9c, H9d |
| 15 | A | `edge_density` | idem | "strong-edge density in [0,1]" — "fraction of `\|grad\| > 16`" (`EDGE_THRESH = 16`) | H9a, H9c, H9d |
| 16 | A | `mean_norm` | idem | `mean / 255` — "DC level" | H9a, H9c, H9d |
| 17 | A | `q_norm` | `x->qindex` (`MACROBLOCK`, `av1/encoder/block.h:966`) | "Quantization index for the current partition block. This is used to as the index to find quantization parameter for luma and chroma transformed coefficients." → `qindex / 255`, "quantization strength" | H9a, H9c, H9d |
| 18 | A | `log_parent_var` | `x->plane[AOM_PLANE_Y].src.buf`, região do pai obtida por aritmética de ponteiros | `log1p(parent var)` — "texture of the containing 2n x 2n block" | H9a, H9c, H9d |
| 19 | A | `parent_contrast` | idem | `(var-pvar)/(var+pvar+1)` — "block-vs-parent texture contrast" | H9a, H9c, H9d |
| 20 | A | `log_sib_mean_var` | idem (três quadrantes irmãos do pai) | `log1p(mean sibling var)` — "texture of the 3 sibling quadrants" | H9a, H9c, H9d |
| 21 | A | `sib_max_contrast` | idem | `(var-maxsib)/(var+maxsib+1)` — "block-vs-worst-sibling contrast" | H9a, H9c, H9d |
| 22 | A | `pos_r` | `blk->mi_row` (`PartitionBlkParams`, `av1/encoder/encodeframe_utils.h:121`) | "Block row and column indices." → `(cell_r * n) / 64`, "vertical position inside the 64px unit" | H9a, H9c, H9d |
| 23 | A | `pos_c` | `blk->mi_col` (`PartitionBlkParams`, `av1/encoder/encodeframe_utils.h:122`) | "Block row and column indices." → `(cell_c * n) / 64`, "horizontal position inside the 64px unit" | H9a, H9c, H9d |
| 24 | B | `has_above` | `xd->above_mbmi` (`MACROBLOCKD`, `av1/common/blockd.h:645`) | "MB_MODE_INFO for 4x4 block above the current block, if up_available == true; otherwise NULL." → 1.0 when the pointer is non-NULL, 0.0 otherwise | H9a, H9c, H9d |
| 25 | B | `has_left` | `xd->left_mbmi` (`MACROBLOCKD`, `av1/common/blockd.h:640`) | "MB_MODE_INFO for 4x4 block to the left of the current block, if left_available == true; otherwise NULL." → 1.0 when the pointer is non-NULL, 0.0 otherwise | H9a, H9c, H9d |
| 26 | B | `above_w_log2` | `mi_size_wide_log2[xd->above_mbmi->bsize]` (`av1/common/common_data.h:25`; `bsize` em `av1/common/blockd.h:228`) | "The Mi_Width_Log2 table in the spec (Section 9.3. Conversion tables)." applied to "The block size of the current coding block" of the above neighbour → falls back to the current block size when the neighbour is missing | H9a, H9c, H9d |
| 27 | B | `above_h_log2` | `mi_size_high_log2[xd->above_mbmi->bsize]` (`av1/common/common_data.h:29`) | "The Mi_Height_Log2 table in the spec (Section 9.3. Conversion tables)." applied to the above neighbour's block size → same fallback | H9a, H9c, H9d |
| 28 | B | `left_w_log2` | `mi_size_wide_log2[xd->left_mbmi->bsize]` | "The Mi_Width_Log2 table in the spec (Section 9.3. Conversion tables)." applied to the left neighbour's block size → same fallback | H9a, H9c, H9d |
| 29 | B | `left_h_log2` | `mi_size_high_log2[xd->left_mbmi->bsize]` | "The Mi_Height_Log2 table in the spec (Section 9.3. Conversion tables)." applied to the left neighbour's block size → same fallback | H9a, H9c, H9d |
| 30 | B | `neigh_finer` | derivado de 24–29 | Fraction of the available neighbours whose block area in log2 mi units (`w_log2 + h_log2`) is smaller than the current node's — relative granularity of the causal neighbourhood | H9a, H9c, H9d |
| 31 | B | `neigh_aniso` | derivado de 24–29 | Mean of `sign(w_log2 - h_log2)` over the available neighbours — anisotropy (elongation direction) of the causal neighbourhood, in [-1,1] | H9a, H9c, H9d |
| 32 | C | `log_dc_q2` | `av1_dc_quant_QTX(x->qindex, 0, xd->bd) >> (xd->bd - 8)` (`av1/common/quant_common.h:45`) | `log1p(dc_q² / 256)` — effective DC dequantisation step for the current `qindex`, rescaled to 8 bits. Note in libaom: "the minimum allowable quantizer is 4; smaller values will underflow to 0 in the actual quantization routines." (`av1/common/quant_common.c:195-196`) | H9a, H9c, H9d |
| 33 | C | `pos_row` | `blk->mi_row` ÷ `cm->height` (`av1/common/av1_common_int.h:783`) | "Block row and column indices." ÷ "Coded frame height" (in mi units) → normalised row position **in the frame** | H9a, H9c, H9d |
| 34 | C | `pos_col` | `blk->mi_col` ÷ `cm->width` (`av1/common/av1_common_int.h:782`) | "Block row and column indices." ÷ "Coded frame width" (in mi units) → normalised column position **in the frame** | H9a, H9c, H9d |
| 35 | C | `depth_log2` | `mi_size_wide_log2[blk->bsize]` (`blk->bsize`: "Block size of current partition.", `av1/encoder/encodeframe_utils.h:142`) | "The Mi_Width_Log2 table in the spec (Section 9.3. Conversion tables)." applied to the current node → node depth, `log2(n/4)` | H9a, H9c, H9d |
| 36 | E | `log_none_rate` | `part_state->this_rdc.rate` (`RD_STATS`, `av1/common/blockd.h:190`; campo em "RD cost for the current block of given partition type.", `av1/encoder/encodeframe_utils.h:184`) | Sem comentário de campo no libaom → `log1p(max(rate, 0))`: rate of the `PARTITION_NONE` candidate already evaluated for this node | H9c, H9d |
| 37 | E | `log_none_dist` | `part_state->this_rdc.dist` (`RD_STATS`, `av1/common/blockd.h:192`) | Sem comentário de campo no libaom → `log1p(max(dist, 0))`: distortion of the evaluated `PARTITION_NONE` candidate | H9c, H9d |
| 38 | E | `log_none_rdcost` | `part_state->this_rdc.rdcost` (`RD_STATS`, `av1/common/blockd.h:198`) | "Please be careful of using rdcost, it's not guaranteed to be set all the time." → `log1p(max(rdcost, 0))`: Lagrangian RD cost of the evaluated `PARTITION_NONE` candidate (o código só consulta o vetor quando `none_rd > 0` e `this_rdc.rate != INT_MAX`, o que garante o campo preenchido) | H9c, H9d |

**Contagem.** 36 linhas para o H9a (índices 0–35) e 39 linhas para o H9c e o H9d
(0–38). Vinte e quatro dos trinta e seis atributos do H9a são descritores de
luminância: o podador implantado é majoritariamente um modelo de pixels e **não
contém grandeza alguma de custo de taxa-distorção**, que só existe no bloco E.

**Custo de obtenção.** Os blocos B e C são leitura grátis de estado já residente
na memória do codificador. O bloco A não exige nova instrumentação, mas exige
cômputo linear na área do bloco. O bloco E exige a avaliação completa de
taxa-distorção do `PARTITION_NONE` — motivo estrutural pelo qual ele é
indisponível ao H9a (`M4` §4.4).

---

## 3. Tabela 2 — Atributos da família avaliados e **não** implantados

Estes atributos pertencem à família H9, foram medidos em portão offline e **não
entram em nenhuma solução implantada**. Constam aqui para que a tabela seja
completa e para que nenhuma leitura futura os confunda com os da Tabela 1.

| Idx | Bloco | Feature name (código) | Origem | Description (English) | Avaliado em |
|--:|:--:|---|---|---|---|
| 36* | D | `log_satd` | `block_satd(block)`, `features.py:252` — luma-fonte apenas, sem predição e sem vizinho | "Sum of \|AC\| Hadamard coefficients of the block (integer-exact rate proxy). Distinct from variance: an L1 measure in the transform domain, sensitive to how energy spreads across frequencies (predictability), not just its total." → `log1p(satd)` | H9b (Gate 2, offline) |
| 37* | D | `satd_l1l2` | idem | `satd / (var * n² + 1)` — SATD normalised by the block's L2 energy | H9b (Gate 2, offline) |
| 36 | B1 | `has_parent` | `dim < 64` | 1.0 if the node has a parent inside the 64×64 unit, else 0.0 | B1 (ablação, offline) |
| 37 | B1 | *(parent `none_rdcost`)* | `RD_STATS.rdcost` do nó-pai | `log1p(parent none_rdcost)`; 0 if no parent / parent missing | B1 (ablação, offline) |
| 38 | B1 | *(parent `none_rate`)* | `RD_STATS.rate` do nó-pai | `log1p(parent none_rate)`; 0 if no parent / parent missing | B1 (ablação, offline) |
| 39 | B1 | *(parent `none_dist`)* | `RD_STATS.dist` do nó-pai | `log1p(parent none_dist)`; 0 if no parent / parent missing | B1 (ablação, offline) |
| 40 | B1 | *(mean sibling `none_rdcost`)* | `RD_STATS.rdcost` dos irmãos anteriores em ordem-z | `log1p(mean none_rdcost of earlier siblings present)`; 0 if none present | B1 (ablação, offline) |
| 41 | B1 | *(sibling count)* | contagem de irmãos anteriores presentes | `(# earlier siblings present) / 3.0` | B1 (ablação, offline) |
| 0 | D' | `pred_avail` | luma-fonte das linhas/colunas adjacentes dentro do superbloco | 1.0 quando os dois vizinhos estão dentro do superbloco (caso contrário o bloco inteiro é neutro, `(0,0,0)`) | D' (crivo de 26/07, offline) |
| 1 | D' | `log_satd_resid` | `features_intrapred.py:98` — melhor predição entre {DC, V, H, PAETH} por SAD | `log1p(SATD_AC do resíduo da melhor predição)` | D' (crivo de 26/07, offline) |
| 2 | D' | `satd_gain` | idem | `(SATD_AC(fonte) - SATD_AC(resíduo)) / (SATD_AC(fonte)+1)` — fração da energia AC removida pela predição a partir dos vizinhos | D' (crivo de 26/07, offline) |

\* No vetor completo de 41 atributos (`H9_SUBSETS["H9c"]`, usado **apenas** na
ablação offline do Gate 2), o bloco D ocupa 36–37 e o bloco E é deslocado para
38–40. O vetor implantado de 39 atributos descarta o bloco D e o E volta a 36–38.

**Por que ficaram de fora.**

- **Bloco D (H9b).** Resultado nulo no portão de sinal. Além disso, a auditoria de
  26/07/2026 estabeleceu que o bloco D **implementado não é o especificado**: o
  plano definia SATD do **resíduo de uma predição intra a partir dos vizinhos
  reconstruídos**, e a implementação calcula o Hadamard do **bloco-fonte**, sem
  predição alguma — estatística exclusivamente da fonte, correlacionada com a
  variância e com os gradientes que o bloco A já contém.
- **Bloco D'.** É a hipótese originalmente especificada, testada depois. Também
  negativa, no nível de 16 px — o mais numeroso —, o que fecha o domínio de
  pixels por uma via a mais.
- **Bloco B1.** Contexto de taxa-distorção hereditário (pai + irmãos anteriores);
  negativo no crivo A5, degrada a decisão ponderada por custo em toda a fronteira.

> **Procedência de §3.** `features.py:193-201` (layout dos blocos), `252-260`
> (`block_satd`), `327-379` (B1); `features_intrapred.py` (D');
> `docs/RESULTADOS_auditoria_dominio_pixels.md` §3 e §6;
> `docs/INVENTARIO_solucoes.md` §2 e §3.5; `M4_atributos_e_politica.md` §4.2.

---

## 4. Resumo por bloco

| Bloco | Atributos | Índices (vetor implantado) | H9a | H9c | H9d | Custo de obtenção |
|---|--:|---|:--:|:--:|:--:|---|
| A — luminância + `q_norm` + posição no superbloco | 24 | 0–23 | ✔ | ✔ | ✔ | cômputo linear na área do bloco; nenhuma instrumentação nova |
| B — vizinhança de particionamento causal | 8 | 24–31 | ✔ | ✔ | ✔ | leitura grátis (`xd->above_mbmi`, `xd->left_mbmi`) |
| C — quantização e posição no quadro | 4 | 32–35 | ✔ | ✔ | ✔ | leitura grátis (`dc_q`, `mi_row`, `mi_col`, `bsize`) |
| D — proxy de SATD | 2 | — | ✘ | ✘ | ✘ | uma transformada de Hadamard por nó |
| E — contexto RD real do `PARTITION_NONE` | 3 | 36–38 | ✘ | ✔ | ✔ | exige a avaliação completa de RD do `PARTITION_NONE` |
| **Total implantado** | | | **36** | **39** | **39** | |
