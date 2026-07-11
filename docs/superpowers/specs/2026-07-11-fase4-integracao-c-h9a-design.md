# Spec — Fase 4: integração em C das features H9a (B/C) + paridade + no-op

**Data:** 2026-07-11. Branch `ml-partition-dev`. Sucede a Fase 3
(`2026-07-11-fase3-estudante-h9a-design.md`, commit `173aa8f`). Contexto:
`docs/PLANO_H9_contribuicao_tese.md` (Fase 4), `docs/ANDAMENTO_tese.md` §4,
`docs/RASTREABILIDADE.md`, `src/aom/CLAUDE.md` (workflow de build).

---

## 1. Objetivo e escopo

Fazer o encoder libaom computar em C **o mesmo vetor de 36 features H9a** que o
estudante Python treinou (Fase 3), e trocar os pesos implantados pelos de 36
features. É engenharia de baixo risco: o sinal já foi provado offline (Gate 3);
aqui garantimos que o C reproduz fielmente o que foi validado e que a mudança é
**inerte quando a flag está desligada**.

**Fora de escopo:**
- Benchmark no encoder real (taxa BD × speedup no teste held-out) — **Fase 5**.
- Qualquer mudança de política/τ (mantém-se a política de 3 ações atual).
- Re-treino/re-export de pesos além do que a Fase 3 já produziu.

## 2. Restrições (Global Constraints)

- **Editar no Windows, compilar/testar no Docker** (`av1_bench`,
  `/workspace/build/venv-ml/bin/python`; builds via `cmake`/`ninja`). Ver
  `src/aom/CLAUDE.md`.
- **`src/aom_baseline` é intocável** (controle cego). Nunca ler/editar.
- **Todo o código novo sob `#if PARTITION_ML_STUDENT`** (default 0). Com a flag
  desligada, o bitstream tem de ser **byte-idêntico** ao `aom_baseline`.
- **`features.node_features_h9a` é a fonte única de verdade** do layout de 36
  features. O C espelha ESSA função; qualquer divergência é bug.
- **Estilo:** clang-format (Google-based), casar o estilo do entorno.
- **Commits:** sem menção a IA/Claude/Co-Authored-By.

## 3. Contrato das features (o que o C deve computar)

Layout (índices 0–35), espelhando `features.node_features_h9(...)[:36]`:
- **A (0–23):** já implementado em `student_node_features`
  (`partition_strategy.c:1758`). **Nenhuma mudança.**
- **B (24–31) — vizinhança:** fontes = `xd->above_mbmi`, `xd->left_mbmi` (idiomas
  nativos já presentes em `partition_strategy.c:626–636`). Convenção de fallback:
  vizinho ausente → usar o **bloco atual** (que é quadrado no nível modelado), i.e.
  `above_bsize = has_above ? xd->above_mbmi->bsize : blk->bsize`.
  - `f[24] = has_above` (`!!xd->above_mbmi`)
  - `f[25] = has_left` (`!!xd->left_mbmi`)
  - `f[26] = mi_size_wide_log2[above_bsize]`, `f[27] = mi_size_high_log2[above_bsize]`
  - `f[28] = mi_size_wide_log2[left_bsize]`, `f[29] = mi_size_high_log2[left_bsize]`
  - `f[30] = neigh_finer`, `f[31] = neigh_aniso` — fórmulas **idênticas** a
    `node_features_h9` (cur_area = cur_w+cur_h com cur = log2(n/4) quadrado;
    `navail = max(has_above+has_left,1)`; `finer = ((has_above && aw+ah<cur_area) +
    (has_left && lw+lh<cur_area))/navail`; `aniso = (has_above*sign(aw-ah) +
    has_left*sign(lw-lh))/navail`). `sign` = `(x>0)-(x<0)`.
- **C (32–35) — quantização/posição:**
  - `f[32] = log1pf(dc_q*dc_q/256)`, `dc_q = av1_dc_quant_QTX(x->qindex,0,xd->bd) >> (xd->bd-8)` (idioma nativo `:623`).
  - `f[33] = mi_row / (frame_h/4.0)`, `f[34] = mi_col / (frame_w/4.0)` — `frame_w/h`
    em pixels (cm->width/cm->height); divisão float, como no Python.
  - `f[35] = mi_size_wide_log2[blk->bsize]` (= log2(n/4), inteiro exato; casa com
    `np.log2(dim//4)` do Python sem erro de float).

**Notas de paridade críticas:** todos os B são inteiros/log2 exatos (paridade
bit-a-bit esperada). `f[30..35]` envolvem float; paridade dentro de ~1–2 ulp. A
posição usa **as mesmas dimensões de quadro** que o dump reporta (§4), então a
comparação é por construção consistente.

## 4. Estratégia de paridade (decisão confirmada)

As features B/C dependem de **estado de runtime do encoder** (decisões de
particionamento dos vizinhos, quantizador) e **não são recomputáveis** só a partir
do quadro-fonte. Logo, o harness passa a comparar a **aritmética** das features
isolada do **sourcing** do ctx:

**Estender o dump do C** (`student_dump_features:1897`) para gravar, além dos 36
features e 3 probs, o **ctx cru** que o C usou. Novo registro:
```
head[10] int32 = { n, mi_row, mi_col, qindex,
                   neigh_avail, above_bsize, left_bsize, dc_q,
                   frame_w, frame_h }
feats[36] float32
probs[3]  float32
```
onde `neigh_avail = has_above | (has_left<<1)` (bit0=above, bit1=left, mesma
convenção do dataset), e `above_bsize`/`left_bsize` são os bsize **resolvidos**
(`has_x ? xd->x_mbmi->bsize : blk->bsize`) — o mesmo idioma nativo. Quando o vizinho
falta, o Python guarda por `has_x` e usa o bloco atual, então o valor gravado é
consistente de qualquer forma.

**`check_feature_parity.py`** passa a: (i) desempacotar `<10i36f3f`; (ii) montar
`ctx = {neigh_avail, above_bsize, left_bsize, dc_q, mi_row, mi_col, frame_w,
frame_h}`; (iii) comparar contra
`features.node_features_h9a(sb_fonte, n, r_cell, c_cell, qindex, ctx)`; (iv) usar
`H9_FEATURE_NAMES[:36]` nos rótulos e `NUM_FEATURES_H9A` no tamanho; (v) re-pontuar
as features do C com o estudante **`student_h9a`** (36 entradas) e comparar as probs.

Isto verifica que, dado ctx + luma idênticos, C e Python computam vetores
idênticos — a garantia exata que precisamos antes de trocar o header.

## 5. Troca do header implantado (acoplado)

Código de 36 features exige header de 36 (`av1_nn_predict` lê `NUM_FEATURES`
entradas). Rodar `export_weights.py --students results/models/student_h9a/students.pt
--out src/aom/av1/encoder/partition_student_weights.h`. O header passa a ter
`AV1_PARTITION_STUDENT_NUM_FEATURES 36`. O `student_real` (24 features) permanece
recuperável no git (`results/models/student_real/` + histórico).

Garantir que a inclusão do header e o uso de `AV1_PARTITION_STUDENT_NUM_FEATURES`
estejam **sob `#if PARTITION_ML_STUDENT`**, para o no-op byte-idêntico valer.

## 6. Gate 4 (nada avança sem passar)

- **Paridade de features (C↔Python):** `check_feature_parity.py` verde nas 36
  features — A e B bit-a-bit (`atol` atual 2e-6), C dentro de ulps.
- **Paridade de probabilidades:** features do C re-pontuadas por `student_h9a`
  casam com as probs do encoder (atol 1e-3, como hoje).
- **No-op byte-idêntico:** build com a flag **desligada** produz md5 de bitstream
  idêntico ao `aom_baseline` (mesma sequência/QP). Prova que a mudança é inerte.
- **Decodificação válida** (aomdec) + `test_libaom` no filtro de partição/intra.

## 7. Builds

- `libaom_ml_check` (RelWithDebInfo, `AOM_TARGET_CPU=generic`, `-DPARTITION_ML_STUDENT=1`)
  — paridade/validade (código C puro, sem SIMD sombreando).
- `libaom_perf` (Release, `-DPARTITION_ML_STUDENT=1`) — timing (para a Fase 5).
- Build da flag-desligada (default) para o no-op vs `aom_baseline`
  (`libaom_perf_anchor`/`aom_baseline`).

## 8. Componentes e arquivos

| Arquivo | Mudança |
|---|---|
| `src/aom/av1/encoder/partition_strategy.c` | `student_node_features`: +B(24–31) +C(32–35) espelhando `node_features_h9a`; `student_dump_features`: estender registro com o ctx cru (head[10]) |
| `src/aom/av1/encoder/partition_student_weights.h` | **gerado** por `export_weights` (36 features); troca o implantado |
| `src/scripts/partition_model/check_feature_parity.py` | 36 features via `node_features_h9a`; desempacota `<10i36f3f`; monta ctx; default `--students` → `student_h9a` |

## 9. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Divergência C↔Python em B/C (fallback, ordem, sign) | `node_features_h9a` é a referência; Gate 4 paridade exata antes de qualquer benchmark |
| No-op quebra (código fora do guard) | Auditar que TODO o novo código + include do header estão sob `#if PARTITION_ML_STUDENT`; md5 vs `aom_baseline` |
| `mi_rows/mi_cols` vs `frame_h/4` em resolução não-múltipla de 4 | Em 4K é exato; o dump reporta o `frame_w/h` usado e o Python usa o mesmo → consistente por construção |
| Disponibilidade de `above/left_mbmi` no ponto de chamada | Já usados pelo código nativo no mesmo caminho RD (causais/disponíveis) |
| Sobrescrever o `student_real` implantado sem retorno | `student_real` versionado no git; header antigo no histórico |

## 10. Entregáveis

1. `student_node_features` de 36 features + dump estendido (C).
2. Header implantado de 36 features (`partition_student_weights.h`).
3. `check_feature_parity.py` estendido, **verde**.
4. Evidência do Gate 4: paridade verde, no-op byte-idêntico (md5), decode válido,
   testes. (Sem benchmark — Fase 5.)
