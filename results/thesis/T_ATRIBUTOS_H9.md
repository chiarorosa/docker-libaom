# Tabela completa dos atributos da família H9

> Materialização, atributo a atributo, do vetor que alimenta os podadores da
> família H9. Os nomes e as descrições estão **em inglês**, na forma exata em que
> aparecem no código que os produz.
>
> A tabela traz duas colunas distintas para o significado de cada atributo:
>
> - **Specification** — a definição formal: a fórmula exata do código e, quando a
>   grandeza é lida de estado do **libaom**, o comentário do libaom reproduzido
>   **literalmente**, entre aspas, antes de qualquer texto derivado (que vem
>   após "→"). É o que garante a rastreabilidade ao código-fonte.
> - **Description** — a explicação didática, em linguagem corrente: o que o
>   atributo mede e por que ele importa para a decisão de particionamento.
>
> Nenhuma especificação foi inventada. Onde o libaom não documenta o campo, isso
> está declarado na própria célula.

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

| Idx | Bloco | Feature name (código) | Origem no libaom (símbolo exato) | Specification | Description | Incluso em |
|--:|:--:|---|---|---|---|---|
| 0 | A | `log_var` | `x->plane[AOM_PLANE_Y].src.buf` — "A buffer containing the source frame." (`av1/encoder/block.h:143`) | `log1p(var)` — "overall texture / flatness" | How much the luma of the whole block varies. A flat block scores low and is a good candidate for a single large partition; a busy block scores high and usually needs to be split. | H9a, H9c, H9d |
| 1 | A | `log_var_q0` | idem (quadrante superior esquerdo do bloco) | `log1p(var)` of quadrant 0 — "per-quadrant texture" | The same texture measure taken on the top-left quarter of the block, so the model can tell whether the detail sits in only part of it. | H9a, H9c, H9d |
| 2 | A | `log_var_q1` | idem (quadrante superior direito) | `log1p(var)` of quadrant 1 — "per-quadrant texture" | Texture of the top-right quarter, same purpose as feature 1. | H9a, H9c, H9d |
| 3 | A | `log_var_q2` | idem (quadrante inferior esquerdo) | `log1p(var)` of quadrant 2 — "per-quadrant texture" | Texture of the bottom-left quarter, same purpose as feature 1. | H9a, H9c, H9d |
| 4 | A | `log_var_q3` | idem (quadrante inferior direito) | `log1p(var)` of quadrant 3 — "per-quadrant texture" | Texture of the bottom-right quarter, same purpose as feature 1. Together, features 1–4 are exactly what a square split would face. | H9a, H9c, H9d |
| 5 | A | `quad_var_spread` | idem | `(max_qv - min_qv)/(max_qv+1)` — "quadrant-variance spread" | How unequal the four quarters are in texture. Near zero means the block is uniformly flat or uniformly busy; near one means one quarter carries all the detail, which argues for splitting. | H9a, H9c, H9d |
| 6 | A | `log_var_qsums` | idem | `log1p(var of quadrant sums)` — "brightness heterogeneity across quadrants" | How different the four quarters are in average brightness. A high value means the block straddles two distinct regions of the picture. | H9a, H9c, H9d |
| 7 | A | `log_var_qvars` | idem | `log1p(var of quadrant vars)` — "texture heterogeneity across quadrants" | How different the four quarters are in amount of detail — whether the detail is evenly spread or concentrated in one corner. | H9a, H9c, H9d |
| 8 | A | `log_hgrad` | idem | `log1p(horizontal grad sum)` — "vertical-edge energy" | Total left-to-right change between neighbouring pixels. It is high when the block contains vertical edges. | H9a, H9c, H9d |
| 9 | A | `log_vgrad` | idem | `log1p(vertical grad sum)` — "horizontal-edge energy" | Total top-to-bottom change between neighbouring pixels. It is high when the block contains horizontal edges. | H9a, H9c, H9d |
| 10 | A | `grad_orient` | idem | `(hgrad - vgrad)/(hgrad+vgrad+1)` — "gradient orientation in [-1,1]" | Which of the two edge energies dominates, on a −1 to +1 scale. It tells the model whether the structure of the block is mostly vertical or mostly horizontal, independently of how strong it is. | H9a, H9c, H9d |
| 11 | A | `log_var_rowsums` | idem | `log1p(var of row sums)` — "horizontal-band structure (HORZ cue)" | How much whole rows differ from one another. A high value means the block is made of horizontal bands, which favours a horizontal partition. | H9a, H9c, H9d |
| 12 | A | `log_var_colsums` | idem | `log1p(var of col sums)` — "vertical-band structure (VERT cue)" | How much whole columns differ from one another — the vertical-band counterpart of feature 11, favouring a vertical partition. | H9a, H9c, H9d |
| 13 | A | `rowcol_orient` | idem | `(vrow - vcol)/(vrow+vcol+1)` — "row-vs-col structure orientation" | The balance between row and column structure on a −1 to +1 scale: a single compact cue for "rectangular horizontal" versus "rectangular vertical". | H9a, H9c, H9d |
| 14 | A | `log_maxgrad` | idem | `log1p(max \|grad\|)` — "strongest edge" | The sharpest single transition anywhere in the block. It separates a block with one hard edge (which a rectangular partition can follow) from a block that is uniformly noisy. | H9a, H9c, H9d |
| 15 | A | `edge_density` | idem | "strong-edge density in [0,1]" — "fraction of `\|grad\| > 16`" (`EDGE_THRESH = 16`) | What share of the block is edge rather than smooth area. One strong edge gives a low density; texture everywhere gives a high one — the two demand different partitions even at equal variance. | H9a, H9c, H9d |
| 16 | A | `mean_norm` | idem | `mean / 255` — "DC level" | The average brightness of the block, on a 0-to-1 scale. It lets the model condition on dark versus bright content, where the same amount of detail is perceived — and coded — differently. | H9a, H9c, H9d |
| 17 | A | `q_norm` | `x->qindex` (`MACROBLOCK`, `av1/encoder/block.h:966`) | "Quantization index for the current partition block. This is used to as the index to find quantization parameter for luma and chroma transformed coefficients." → `qindex / 255`, "quantization strength" | How coarse the quantisation is, on a 0-to-1 scale. It is the single most important non-pixel input: at coarse quantisation fine detail is discarded anyway, so large partitions become cheap and the same block should be split less. | H9a, H9c, H9d |
| 18 | A | `log_parent_var` | `x->plane[AOM_PLANE_Y].src.buf`, região do pai obtida por aritmética de ponteiros | `log1p(parent var)` — "texture of the containing 2n x 2n block" | The texture of the twice-as-large block that contains this node. It tells the model whether the node sits inside a calm or a busy neighbourhood, which the node alone cannot reveal. | H9a, H9c, H9d |
| 19 | A | `parent_contrast` | idem | `(var-pvar)/(var+pvar+1)` — "block-vs-parent texture contrast" | Whether this node is busier or calmer than its parent. A node far calmer than its surroundings is a strong candidate for `PARTITION_NONE`, even if its absolute texture is not low. | H9a, H9c, H9d |
| 20 | A | `log_sib_mean_var` | idem (três quadrantes irmãos do pai) | `log1p(mean sibling var)` — "texture of the 3 sibling quadrants" | The average texture of the three sibling quarters that share the same parent block. | H9a, H9c, H9d |
| 21 | A | `sib_max_contrast` | idem | `(var-maxsib)/(var+maxsib+1)` — "block-vs-worst-sibling contrast" | How this node compares with its busiest sibling. It distinguishes "the whole parent region is busy" from "only the block next to me is busy", two situations that call for different decisions. | H9a, H9c, H9d |
| 22 | A | `pos_r` | `blk->mi_row` (`PartitionBlkParams`, `av1/encoder/encodeframe_utils.h:121`) | "Block row and column indices." → `(cell_r * n) / 64`, "vertical position inside the 64px unit" | Where the node sits vertically inside its 64-pixel superblock, on a 0-to-1 scale. It is a geometric address that lets the model recognise the border nodes, whose causal context is truncated. | H9a, H9c, H9d |
| 23 | A | `pos_c` | `blk->mi_col` (`PartitionBlkParams`, `av1/encoder/encodeframe_utils.h:122`) | "Block row and column indices." → `(cell_c * n) / 64`, "horizontal position inside the 64px unit" | Where the node sits horizontally inside its 64-pixel superblock, same purpose as feature 22. | H9a, H9c, H9d |
| 24 | B | `has_above` | `xd->above_mbmi` (`MACROBLOCKD`, `av1/common/blockd.h:645`) | "MB_MODE_INFO for 4x4 block above the current block, if up_available == true; otherwise NULL." → 1.0 when the pointer is non-NULL, 0.0 otherwise | Whether there is an already-coded block above this one. When there is not — at the top of the frame or tile — the neighbourhood cues that follow are meaningless, and this flag is what tells the model to disregard them. | H9a, H9c, H9d |
| 25 | B | `has_left` | `xd->left_mbmi` (`MACROBLOCKD`, `av1/common/blockd.h:640`) | "MB_MODE_INFO for 4x4 block to the left of the current block, if left_available == true; otherwise NULL." → 1.0 when the pointer is non-NULL, 0.0 otherwise | Whether there is an already-coded block to the left, with the same role as feature 24. | H9a, H9c, H9d |
| 26 | B | `above_w_log2` | `mi_size_wide_log2[xd->above_mbmi->bsize]` (`av1/common/common_data.h:25`; `bsize` em `av1/common/blockd.h:228`) | "The Mi_Width_Log2 table in the spec (Section 9.3. Conversion tables)." applied to "The block size of the current coding block" of the above neighbour → falls back to the current block size when the neighbour is missing | How wide the block above ended up being, after the encoder actually decided it. A large neighbour is evidence that the region is smooth and that a large partition will also work here. | H9a, H9c, H9d |
| 27 | B | `above_h_log2` | `mi_size_high_log2[xd->above_mbmi->bsize]` (`av1/common/common_data.h:29`) | "The Mi_Height_Log2 table in the spec (Section 9.3. Conversion tables)." applied to the above neighbour's block size → same fallback | How tall the block above ended up being. Together with feature 26 it gives both the size and the shape of the decision taken just above this node. | H9a, H9c, H9d |
| 28 | B | `left_w_log2` | `mi_size_wide_log2[xd->left_mbmi->bsize]` | "The Mi_Width_Log2 table in the spec (Section 9.3. Conversion tables)." applied to the left neighbour's block size → same fallback | How wide the block to the left ended up being, same reasoning as feature 26. | H9a, H9c, H9d |
| 29 | B | `left_h_log2` | `mi_size_high_log2[xd->left_mbmi->bsize]` | "The Mi_Height_Log2 table in the spec (Section 9.3. Conversion tables)." applied to the left neighbour's block size → same fallback | How tall the block to the left ended up being, same reasoning as feature 27. | H9a, H9c, H9d |
| 30 | B | `neigh_finer` | derivado de 24–29 | Fraction of the available neighbours whose block area in log2 mi units (`w_log2 + h_log2`) is smaller than the current node's — relative granularity of the causal neighbourhood | Whether the already-coded neighbours were cut finer than the size being considered here. It is the most direct hint available before the search that this region of the picture demands small partitions. | H9a, H9c, H9d |
| 31 | B | `neigh_aniso` | derivado de 24–29 | Mean of `sign(w_log2 - h_log2)` over the available neighbours — anisotropy (elongation direction) of the causal neighbourhood, in [-1,1] | Whether the neighbouring blocks came out elongated, and in which direction. Neighbours that are all wide and short suggest that a horizontal partition will win here too. | H9a, H9c, H9d |
| 32 | C | `log_dc_q2` | `av1_dc_quant_QTX(x->qindex, 0, xd->bd) >> (xd->bd - 8)` (`av1/common/quant_common.h:45`) | `log1p(dc_q² / 256)` — effective DC dequantisation step for the current `qindex`, rescaled to 8 bits. Note in libaom: "the minimum allowable quantizer is 4; smaller values will underflow to 0 in the actual quantization routines." (`av1/common/quant_common.c:195-196`) | The actual size of the quantisation step applied to the block's DC coefficient. It is the physical quantity behind the `qindex` of feature 17: the model reacts to how coarse the encoding really is, not to a table index whose relation to coarseness is non-linear. | H9a, H9c, H9d |
| 33 | C | `pos_row` | `blk->mi_row` ÷ `cm->height` (`av1/common/av1_common_int.h:783`) | "Block row and column indices." ÷ "Coded frame height" (in mi units) → normalised row position **in the frame** | Where the node sits vertically **in the whole frame**, on a 0-to-1 scale — distinct from feature 22, which locates it inside the superblock. It captures content that varies with position (sky at the top, ground at the bottom) and identifies the nodes at the frame border. | H9a, H9c, H9d |
| 34 | C | `pos_col` | `blk->mi_col` ÷ `cm->width` (`av1/common/av1_common_int.h:782`) | "Block row and column indices." ÷ "Coded frame width" (in mi units) → normalised column position **in the frame** | Where the node sits horizontally in the whole frame, same purpose as feature 33. | H9a, H9c, H9d |
| 35 | C | `depth_log2` | `mi_size_wide_log2[blk->bsize]` (`blk->bsize`: "Block size of current partition.", `av1/encoder/encodeframe_utils.h:142`) | "The Mi_Width_Log2 table in the spec (Section 9.3. Conversion tables)." applied to the current node → node depth, `log2(n/4)` | How large the node is, and therefore how deep it lies in the partition tree. It is what allows one and the same feature vector to describe a 64-pixel node and a 16-pixel one without ambiguity. | H9a, H9c, H9d |
| 36 | E | `log_none_rate` | `part_state->this_rdc.rate` (`RD_STATS`, `av1/common/blockd.h:190`; campo em "RD cost for the current block of given partition type.", `av1/encoder/encodeframe_utils.h:184`) | Sem comentário de campo no libaom → `log1p(max(rate, 0))`: rate of the `PARTITION_NONE` candidate already evaluated for this node | How many bits it actually cost to code this block as a single unit — measured by the encoder, not estimated. An expensive `PARTITION_NONE` is direct evidence that the block should be split. | H9c, H9d |
| 37 | E | `log_none_dist` | `part_state->this_rdc.dist` (`RD_STATS`, `av1/common/blockd.h:192`) | Sem comentário de campo no libaom → `log1p(max(dist, 0))`: distortion of the evaluated `PARTITION_NONE` candidate | How much error was left behind when the block was coded as a single unit. High distortion means one large block failed to represent the content. | H9c, H9d |
| 38 | E | `log_none_rdcost` | `part_state->this_rdc.rdcost` (`RD_STATS`, `av1/common/blockd.h:198`) | "Please be careful of using rdcost, it's not guaranteed to be set all the time." → `log1p(max(rdcost, 0))`: Lagrangian RD cost of the evaluated `PARTITION_NONE` candidate (o código só consulta o vetor quando `none_rd > 0` e `this_rdc.rate != INT_MAX`, o que garante o campo preenchido) | The combined price in bits and quality of coding the block as a single unit — the one number the encoder compares against every other partition. It summarises how good the "do not split" answer already is, and it is the information no pre-search pruner can have. | H9c, H9d |

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

| Idx | Bloco | Feature name (código) | Origem | Specification | Description | Avaliado em |
|--:|:--:|---|---|---|---|---|
| 36* | D | `log_satd` | `block_satd(block)`, `features.py:252` — luma-fonte apenas, sem predição e sem vizinho | "Sum of \|AC\| Hadamard coefficients of the block (integer-exact rate proxy). Distinct from variance: an L1 measure in the transform domain, sensitive to how energy spreads across frequencies (predictability), not just its total." → `log1p(satd)` | How much energy the block carries once transformed, ignoring its average level — a cheap stand-in for "how many bits will this cost". In practice it mostly repeats what the variance and the gradients of block A already say. | H9b (Gate 2, offline) |
| 37* | D | `satd_l1l2` | idem | `satd / (var * n² + 1)` — SATD normalised by the block's L2 energy | The same measure divided by the block's own energy, so that absolute texture cancels out. It was meant to isolate predictability, but is still computed from the source block alone. | H9b (Gate 2, offline) |
| 36 | B1 | `has_parent` | `dim < 64` | 1.0 if the node has a parent inside the 64×64 unit, else 0.0 | Whether this node has a parent block inside the superblock — false only at the 64-pixel level, where the features that follow have nothing to read. | B1 (ablação, offline) |
| 37 | B1 | *(parent `none_rdcost`)* | `RD_STATS.rdcost` do nó-pai | `log1p(parent none_rdcost)`; 0 if no parent / parent missing | How expensive it was to code the parent block as a single unit. The parent is always evaluated before the encoder recurses into its children, so this is legitimately available before this node is searched. | B1 (ablação, offline) |
| 38 | B1 | *(parent `none_rate`)* | `RD_STATS.rate` do nó-pai | `log1p(parent none_rate)`; 0 if no parent / parent missing | How many bits the parent block took as a single unit — the rate half of the inherited cost. | B1 (ablação, offline) |
| 39 | B1 | *(parent `none_dist`)* | `RD_STATS.dist` do nó-pai | `log1p(parent none_dist)`; 0 if no parent / parent missing | How much error the parent block left as a single unit — the distortion half of the inherited cost. | B1 (ablação, offline) |
| 40 | B1 | *(mean sibling `none_rdcost`)* | `RD_STATS.rdcost` dos irmãos anteriores em ordem-z | `log1p(mean none_rdcost of earlier siblings present)`; 0 if none present | The average cost of the sibling quadrants already searched before this one, in coding order — how expensive this corner of the picture has been proving so far. | B1 (ablação, offline) |
| 41 | B1 | *(sibling count)* | contagem de irmãos anteriores presentes | `(# earlier siblings present) / 3.0` | How many siblings had already been searched when this node was scored, which says how much confidence the previous feature deserves. | B1 (ablação, offline) |
| 0 | D' | `pred_avail` | luma-fonte das linhas/colunas adjacentes dentro do superbloco | 1.0 quando os dois vizinhos estão dentro do superbloco (caso contrário o bloco inteiro é neutro, `(0,0,0)`) | Whether the row above and the column to the left are both available, so that an intra prediction can actually be built. When they are not, the two features that follow are set to zero. | D' (crivo de 26/07, offline) |
| 1 | D' | `log_satd_resid` | `features_intrapred.py:98` — melhor predição entre {DC, V, H, PAETH} por SAD | `log1p(SATD_AC do resíduo da melhor predição)` | How much energy is left over after predicting the block from its neighbours with the cheapest AV1 intra modes — that is, what the encoder would actually still have to code. | D' (crivo de 26/07, offline) |
| 2 | D' | `satd_gain` | idem | `(SATD_AC(fonte) - SATD_AC(resíduo)) / (SATD_AC(fonte)+1)` — fração da energia AC removida pela predição a partir dos vizinhos | What share of the block's energy the prediction manages to remove. This is predictability proper — the property that variance cannot capture, since a busy but perfectly predictable block is cheap to code. | D' (crivo de 26/07, offline) |

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
