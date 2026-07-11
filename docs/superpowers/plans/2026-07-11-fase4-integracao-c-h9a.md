# Fase 4 — Integração em C das features H9a — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fazer o encoder libaom computar em C o vetor de 36 features H9a (espelhando `features.node_features_h9a`), trocar o header implantado para 36 features, e provar por paridade C↔Python + no-op byte-idêntico que a troca é fiel e inerte quando desligada.

**Architecture:** Estende `student_node_features` (`partition_strategy.c`) com os blocos B (vizinhança) e C (quant/posição), reusando os idiomas nativos de sourcing já presentes no arquivo. Estende o dump de paridade para carregar o `ctx` cru, e o harness Python compara via `node_features_h9a`. Tudo sob `#if PARTITION_ML_STUDENT`.

**Tech Stack:** C (libaom v3.10.0), Python (parity harness). Build/test **exclusivamente no container** `av1_bench` (cmake+ninja+ccache); edição no Windows.

## Global Constraints

- **Editar no Windows, compilar/testar no Docker** `av1_bench`. Rebuild: `docker exec av1_bench bash -lc 'cmake --build /workspace/build/<dir> --target aomenc -j"$(nproc)"'`. Python: `/workspace/build/venv-ml/bin/python`.
- **`src/aom_baseline` é INTOCÁVEL** (controle cego; nunca ler/editar).
- **Todo código novo sob `#if PARTITION_ML_STUDENT`** (a região é `partition_strategy.c:1551–1992`; o `#include` do header já está em `:1556`). Flag off ⇒ bitstream byte-idêntico ao `aom_baseline`.
- **`features.node_features_h9a` é a fonte única de verdade** do layout de 36 features. O C espelha ESSA função. Divergência = bug.
- **Bundle/pesos:** `results/models/student_h9a/students.pt` (36 features, Fase 3).
- **Estilo:** clang-format Google-based, casar o entorno. **Commits:** sem menção a IA/Claude/Co-Authored-By. Branch `ml-partition-dev` (commit aqui; não tocar main).
- Sem suíte pytest: verificação = builds + smokes no container + o harness de paridade + o md5 do no-op.

**Record de dump (contrato entre C e Python):**
`head[10] int32 = {n, mi_row, mi_col, qindex, neigh_avail, above_bsize, left_bsize, dc_q, frame_w, frame_h}` ++ `feats[36] float32` ++ `probs[3] float32`. Tamanho = 10·4 + 36·4 + 3·4 = **196 bytes**. `neigh_avail = has_above | (has_left<<1)`.

---

### Task 1: C — features B/C + dump estendido + header de 36

Estende `student_node_features` com B(24–31)+C(32–35) espelhando `node_features_h9a`, threading `cm` para as dimensões de quadro; estende `student_dump_features` para o `head[10]`; troca o header implantado para 36. **Ordem importa:** o header (36) vai primeiro, senão o array `feats[AV1_PARTITION_STUDENT_NUM_FEATURES]` (24) transborda.

**Files:**
- Modify: `src/aom/av1/encoder/partition_strategy.c:1758` (`student_node_features`), `:1897` (`student_dump_features`), `:1977` e `:1981` (call sites)
- Regenerate: `src/aom/av1/encoder/partition_student_weights.h` (via `export_weights.py`)

**Interfaces:**
- Consumes: `features.node_features_h9a` (referência de fórmulas); `results/models/student_h9a/students.pt`.
- Produces: `student_node_features(const AV1_COMMON *cm, const MACROBLOCK *x, const PartitionBlkParams *blk, float *feats)` — preenche `feats[0..35]`; `student_dump_features(const AV1_COMMON *cm, const MACROBLOCK *x, const PartitionBlkParams *blk, int qindex, const float *feats, const float *probs)` — grava o record de 196 B. Consumido pela Task 2 (harness).

- [ ] **Step 1: Exportar o header de 36 features (deve preceder o código C)**

Run:
```bash
docker exec av1_bench bash -lc 'cd /workspace && build/venv-ml/bin/python \
  src/scripts/partition_model/export_weights.py \
  --students results/models/student_h9a/students.pt \
  --out src/aom/av1/encoder/partition_student_weights.h && \
  grep -m1 AV1_PARTITION_STUDENT_NUM_FEATURES src/aom/av1/encoder/partition_student_weights.h'
```
Expected: `#define AV1_PARTITION_STUDENT_NUM_FEATURES 36`.

- [ ] **Step 2: Adicionar os blocos B/C em `student_node_features`**

Em `partition_strategy.c`, trocar a assinatura (linha 1758):
```c
static void student_node_features(const AV1_COMMON *cm, const MACROBLOCK *x,
                                  const PartitionBlkParams *blk, float *feats) {
```
E, imediatamente antes do `}` que fecha a função (após `feats[23] = ...`, hoje linha 1820), inserir:
```c

  // --- H9a block B: neighbor partition context (mirrors node_features_h9). ---
  // Missing neighbor -> current (square) block, matching the Python fallback.
  const BLOCK_SIZE bsz = blk->bsize;
  const int has_above = !!xd->above_mbmi;
  const int has_left = !!xd->left_mbmi;
  const BLOCK_SIZE above_bsize = has_above ? xd->above_mbmi->bsize : bsz;
  const BLOCK_SIZE left_bsize = has_left ? xd->left_mbmi->bsize : bsz;
  const int aw = mi_size_wide_log2[above_bsize];
  const int ah = mi_size_high_log2[above_bsize];
  const int lw = mi_size_wide_log2[left_bsize];
  const int lh = mi_size_high_log2[left_bsize];
  const int cur_wh = mi_size_wide_log2[bsz];   // square at modeled levels
  const int cur_area = cur_wh + cur_wh;
  const int navail = (has_above + has_left) > 0 ? (has_above + has_left) : 1;
  const double finer =
      (double)((has_above && (aw + ah < cur_area)) +
               (has_left && (lw + lh < cur_area))) / navail;
  const double aniso =
      (double)((has_above ? ((aw > ah) - (aw < ah)) : 0) +
               (has_left ? ((lw > lh) - (lw < lh)) : 0)) / navail;
  feats[24] = (float)has_above;
  feats[25] = (float)has_left;
  feats[26] = (float)aw;
  feats[27] = (float)ah;
  feats[28] = (float)lw;
  feats[29] = (float)lh;
  feats[30] = (float)finer;
  feats[31] = (float)aniso;

  // --- H9a block C: quantization / position. ---
  const int dc_q = av1_dc_quant_QTX(x->qindex, 0, xd->bd) >> (xd->bd - 8);
  feats[32] = (float)log1p((double)(dc_q * dc_q) / 256.0);
  const int frame_w = cm->width, frame_h = cm->height;
  feats[33] = frame_h > 0 ? (float)((double)blk->mi_row / (frame_h / 4.0)) : 0.0f;
  feats[34] = frame_w > 0 ? (float)((double)blk->mi_col / (frame_w / 4.0)) : 0.0f;
  feats[35] = (float)mi_size_wide_log2[bsz];
```

- [ ] **Step 3: Estender `student_dump_features` para o record de 196 B**

Substituir a função inteira (`partition_strategy.c:1897–1912`) por:
```c
static void student_dump_features(const AV1_COMMON *cm, const MACROBLOCK *x,
                                  const PartitionBlkParams *blk, int qindex,
                                  const float *feats, const float *probs) {
  static FILE *fp = NULL;
  static int checked = 0;
  if (!checked) {
    const char *path = getenv("AV1_STUDENT_FEATURE_DUMP");
    if (path) fp = fopen(path, "ab");
    checked = 1;
  }
  if (!fp) return;
  const MACROBLOCKD *const xd = &x->e_mbd;
  const int has_above = !!xd->above_mbmi;
  const int has_left = !!xd->left_mbmi;
  const int neigh_avail = has_above | (has_left << 1);
  const int above_bsize = has_above ? xd->above_mbmi->bsize : blk->bsize;
  const int left_bsize = has_left ? xd->left_mbmi->bsize : blk->bsize;
  const int dc_q = av1_dc_quant_QTX(x->qindex, 0, xd->bd) >> (xd->bd - 8);
  const int32_t head[10] = { block_size_wide[blk->bsize], blk->mi_row,
                             blk->mi_col, qindex, neigh_avail, above_bsize,
                             left_bsize, dc_q, cm->width, cm->height };
  fwrite(head, sizeof(head), 1, fp);
  fwrite(feats, sizeof(float), AV1_PARTITION_STUDENT_NUM_FEATURES, fp);
  fwrite(probs, sizeof(float), 3, fp);
}
```

- [ ] **Step 4: Atualizar os call sites (passar `cm`)**

Em `student_prune_partition` (que tem `const AV1_COMMON *cm`), trocar a chamada (linha 1977):
```c
    student_node_features(cm, x, blk, feats);
```
E a do dump (linha 1981):
```c
    student_dump_features(cm, x, blk, x->qindex, feats, probs);
```

- [ ] **Step 5: Build (flag on) — compila limpo**

Run:
```bash
docker exec av1_bench bash -lc 'cmake --build /workspace/build/libaom_ml_check \
  --target aomenc -j"$(nproc)" 2>&1 | tail -5'
```
Expected: build conclui sem erro (`Linking C executable aomenc` / sem `error:`).

- [ ] **Step 6: Smoke — dump com record de 196 B + encode válido**

Run:
```bash
docker exec av1_bench bash -lc 'cd /workspace &&
build/venv-ml/bin/python - <<PY
import numpy as np
h,w=256,448
rng=np.random.default_rng(0); yy,xx=np.mgrid[0:h,0:w]
y=(96+64*np.sin(xx/17.)*np.cos(yy/23.)+rng.normal(0,24,(h,w)))
y[:,w//3:w//3+8]=235; y[h//2:h//2+6,:]=16
y=np.clip(y,0,255).astype("uint8"); uv=np.full((h//2,w//2),128,"uint8")
open("/tmp/f4.yuv","wb").write(y.tobytes()+uv.tobytes()+uv.tobytes())
PY
rm -f /tmp/f4dump.bin
AV1_STUDENT_FEATURE_DUMP=/tmp/f4dump.bin build/libaom_ml_check/aomenc \
  --usage=2 --passes=1 --threads=1 --cpu-used=0 --end-usage=q --cq-level=32 \
  -w 448 -h 256 --limit=1 -o /tmp/f4.ivf /tmp/f4.yuv 2>/dev/null &&
python3 -c "import os;sz=os.path.getsize(\"/tmp/f4dump.bin\");print(\"dump bytes\",sz,\"rec196 multiple:\",sz%196==0,\"recs:\",sz//196)"'
```
Expected: encode OK; `rec196 multiple: True` e `recs: > 0`.

- [ ] **Step 7: Commit**

```bash
git add src/aom/av1/encoder/partition_strategy.c src/aom/av1/encoder/partition_student_weights.h
git commit -F- <<'EOF'
Fase 4: compute H9a features B/C in C + extend parity dump

student_node_features now fills the 36-feature H9a vector (adds neighbor
partition context and quant/position, mirroring features.node_features_h9a),
and the parity dump carries the raw ctx (head[10]) so Python can compare via
node_features_h9a. Deployed header regenerated to 36 features (student_h9a).
All under PARTITION_ML_STUDENT.
EOF
```

---

### Task 2: Python — harness de paridade 36 features + Gate de paridade

Estende `check_feature_parity.py` para 36 features via `node_features_h9a`, desempacotando o `head[10]` e montando o `ctx`; roda o gate de paridade (deve ficar VERDE).

**Files:**
- Modify: `src/scripts/partition_model/check_feature_parity.py`

**Interfaces:**
- Consumes: dump da Task 1 (record 196 B); `features.node_features_h9a`, `features.H9_FEATURE_NAMES`, `features.NUM_FEATURES_H9A`; `results/models/student_h9a/students.pt`.
- Produces: `PARITY OK` (features + probs) contra o build `libaom_ml_check`.

- [ ] **Step 1: Rodar o harness atual e confirmar que quebra (red)**

Run:
```bash
docker exec av1_bench bash -lc 'cd /workspace && build/venv-ml/bin/python \
  src/scripts/partition_model/check_feature_parity.py \
  --aomenc build/libaom_ml_check/aomenc \
  --students results/models/student_h9a/students.pt 2>&1 | tail -5'
```
Expected: FALHA — o record agora tem 196 B mas o harness espera `<4i{24+3}f` (108 B), então `dump size not a multiple of record size` (ou mismatch de features).

- [ ] **Step 2: Atualizar o record, o ctx e a referência para 36 features**

Trocar a linha do struct (`check_feature_parity.py:35`):
```python
# head[10] ++ feats[N] ++ probs[3]
REC = struct.Struct("<10i{}f".format(featmod.NUM_FEATURES_H9A + 3))
```
Trocar o default de `--students` (linhas 72-74) para o estudante H9a:
```python
    p.add_argument("--students", default="/workspace/results/models/"
                                        "student_h9a/students.pt",
                   help="if set, also verify C probs vs this PyTorch student")
```
No bloco que carrega o estudante (linhas 116-119), usar a largura do bundle:
```python
        nfeat = bundle.get("num_features", featmod.NUM_FEATURES_H9A)
        for dim in bundle["students"]:
            net = studentmod.make_student(nfeat, bundle["hidden"])
            net.load_state_dict(bundle["students"][dim])
            nets[dim] = net.eval()
```
Trocar os `worst`/loop (linhas 121-149) para desempacotar o `head[10]`, montar o ctx e comparar via `node_features_h9a`:
```python
    NF = featmod.NUM_FEATURES_H9A
    worst = np.zeros(NF)
    worst_prob = np.zeros(3)
    seen = set()
    checked = 0
    for i in range(n_rec):
        vals = REC.unpack_from(blob, i * REC.size)
        (n, mi_row, mi_col, qindex, neigh_avail, above_bsize, left_bsize,
         dc_q, frame_w, frame_h) = vals[:10]
        cf = np.array(vals[10:10 + NF], dtype=np.float32)
        cprobs = np.array(vals[10 + NF:], dtype=np.float32)
        key = (n, mi_row, mi_col)
        if key in seen:
            continue  # rd_pick_partition may revisit a node; one check suffices
        seen.add(key)
        sb_r, sb_c = (mi_row & ~15) * 4, (mi_col & ~15) * 4
        sb = y[sb_r:sb_r + 64, sb_c:sb_c + 64]
        r_cell = (mi_row & 15) // (n // 4)
        c_cell = (mi_col & 15) // (n // 4)
        ctx = {"neigh_avail": neigh_avail, "above_bsize": above_bsize,
               "left_bsize": left_bsize, "dc_q": dc_q, "mi_row": mi_row,
               "mi_col": mi_col, "frame_w": frame_w, "frame_h": frame_h}
        pf = featmod.node_features_h9a(sb, n, r_cell, c_cell, qindex, ctx)
        worst = np.maximum(worst, np.abs(cf.astype(np.float64) -
                                         pf.astype(np.float64)))
        if nets is not None and n in nets:
            import torch
            import torch.nn.functional as F
            with torch.no_grad():
                logits = nets[n](torch.tensor(pf, dtype=torch.float32)[None])
                pprobs = F.softmax(logits, dim=-1)[0].numpy()
            worst_prob = np.maximum(worst_prob, np.abs(
                cprobs.astype(np.float64) - pprobs.astype(np.float64)))
        checked += 1
```
E os nomes/contagem no relatório (linhas 151-159):
```python
    print("checked {} unique nodes ({} records)".format(checked, n_rec))
    bad = 0
    names = featmod.H9_FEATURE_NAMES
    for k in range(featmod.NUM_FEATURES_H9A):
        flag = ""
        if worst[k] > args.atol:
            flag = "  <-- MISMATCH"
            bad += 1
        print("  [{:>2}] {:<18} max|dC-dPy| = {:.3e}{}".format(
            k, names[k], worst[k], flag))
```

- [ ] **Step 3: Rodar o gate de paridade (green)**

Run:
```bash
docker exec av1_bench bash -lc 'cd /workspace && build/venv-ml/bin/python \
  src/scripts/partition_model/check_feature_parity.py \
  --aomenc build/libaom_ml_check/aomenc \
  --students results/models/student_h9a/students.pt 2>&1 | tail -45'
```
Expected: 36 linhas de feature, todas sem `MISMATCH` (A/B bit-a-bit ~0, C ≤ ~1e-6); bloco de probs sem `MISMATCH`; última linha `PARITY OK (atol=2e-06)`.

Se alguma feature 24–35 der MISMATCH, é divergência C↔Python — corrigir o C da Task 1 (a referência é `node_features_h9a`) e re-rodar. Se persistir, reportar BLOCKED com a tabela de deltas.

- [ ] **Step 4: Commit**

```bash
git add src/scripts/partition_model/check_feature_parity.py
git commit -F- <<'EOF'
Fase 4: extend parity harness to the 36-feature H9a vector

Unpacks the extended dump (head[10] with the raw RD context), rebuilds the
ctx, and compares against features.node_features_h9a; scores probs with the
student_h9a bundle. Parity green on all 36 features.
EOF
```

---

### Task 3: Gate 4 — no-op byte-idêntico + testes

Prova que a flag desligada é byte-idêntica ao `aom_baseline` (a segurança central) e roda a sanidade de testes.

**Files:** nenhum (verificação). Cria o build `libaom_noop` se não existir.

**Interfaces:** Consumes Tasks 1-2. Produces evidência do Gate 4.

- [ ] **Step 1: Configurar `libaom_noop` (src/aom, flag OFF, config do anchor)**

Run (idempotente; pula se já configurado):
```bash
docker exec av1_bench bash -lc 'test -f /workspace/build/libaom_noop/CMakeCache.txt || \
  cmake -S /workspace/src/aom -B /workspace/build/libaom_noop -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCONFIG_INTERNAL_STATS=0 -DENABLE_TESTS=OFF \
    -DENABLE_EXAMPLES=ON -DENABLE_DOCS=OFF -DENABLE_CCACHE=1 2>&1 | tail -3'
```
Expected: `Generating done` / `Build files have been written` (ou nada, se já existia).

- [ ] **Step 2: Build `libaom_noop` (flag off)**

Run:
```bash
docker exec av1_bench bash -lc 'cmake --build /workspace/build/libaom_noop \
  --target aomenc -j"$(nproc)" 2>&1 | tail -4'
```
Expected: build OK.

- [ ] **Step 3: No-op byte-idêntico vs `aom_baseline` (anchor)**

Run (mesma sequência/params nos dois; compara md5 do bitstream):
```bash
docker exec av1_bench bash -lc 'cd /workspace &&
test -f /tmp/f4.yuv || dd if=/dev/zero of=/tmp/f4.yuv bs=1 count=$((448*256*3/2)) 2>/dev/null
for b in libaom_noop libaom_perf_anchor; do
  build/$b/aomenc --usage=2 --passes=1 --threads=1 --cpu-used=0 --end-usage=q \
    --cq-level=32 -w 448 -h 256 --limit=1 -o /tmp/noop_$b.ivf /tmp/f4.yuv 2>/dev/null
done
md5sum /tmp/noop_libaom_noop.ivf /tmp/noop_libaom_perf_anchor.ivf'
```
Expected: os dois md5 **idênticos** (prova que a mudança guardada é inerte com a flag off).

- [ ] **Step 4: Decodificação válida do stream flag-on**

Run (constrói o `aomdec` do build baseline e decodifica o stream flag-on — se um decoder baseline lê nosso stream, ele é conforme):
```bash
docker exec av1_bench bash -lc 'cd /workspace &&
cmake --build build/libaom_perf_anchor --target aomdec -j"$(nproc)" >/dev/null 2>&1 &&
build/libaom_perf_anchor/aomdec -o /tmp/f4_dec.y4m /tmp/f4.ivf 2>&1 | tail -2 &&
echo "decoded bytes:" $(stat -c%s /tmp/f4_dec.y4m 2>/dev/null || echo MISSING)'
```
Expected: decodifica sem erro; arquivo de saída não-vazio. (`/tmp/f4.ivf` foi gerado pelo build flag-on na Task 1.)

- [ ] **Step 5: Sanidade de testes (encoder não quebrou)**

Run:
```bash
docker exec av1_bench bash -lc 'cmake --build /workspace/build/libaom_dev_generic \
  --target test_libaom -j"$(nproc)" >/dev/null 2>&1 &&
/workspace/build/libaom_dev_generic/test_libaom \
  --gtest_filter="EncodeAPI.AllIntra*:KeyValAPI.*partition*:KeyValAPI.*intra*" \
  2>&1 | tail -6'
```
Expected: `[  PASSED  ] N tests.`, zero falhas. (`libaom_dev_generic` é flag-off; valida que a mudança guardada não regrediu o encoder base.)

- [ ] **Step 6: Registrar evidência do Gate 4**

Run:
```bash
docker exec av1_bench bash -lc 'cd /workspace && {
  echo "Gate 4 evidence ($(date -u +%FT%TZ)):"
  echo "no-op md5 (noop vs anchor):"; md5sum /tmp/noop_*.ivf
} > results/models/student_h9a/gate4_evidence.txt && cat results/models/student_h9a/gate4_evidence.txt'
git add -f results/models/student_h9a/gate4_evidence.txt
git commit -F- <<'EOF'
Fase 4: Gate 4 evidence (no-op byte-identical, decode valid, tests pass)

Flag-off src/aom build is byte-identical to aom_baseline (md5 match), the
flag-on stream decodes cleanly, and the scoped unit tests pass. C<->Python
feature+prob parity is green (Task 2). H9a is now the deployed student.
EOF
```

---

## Notas de execução

- **Ordem dentro da Task 1 é obrigatória:** header de 36 (Step 1) antes de escrever `feats[24..35]` (Step 2), senão o array de 24 transborda.
- **Se a paridade (Task 2 Step 3) reprovar em C (32–35):** revisar `frame_w/h` (usar `cm->width/height`) e o `dc_q >> (bd-8)`; em B (24–31): o fallback de vizinho ausente (`blk->bsize`) e o sinal de `aniso`.
- **No-op é a trava de segurança:** se os md5 diferirem, algum código novo vazou para fora do `#if PARTITION_ML_STUDENT` — auditar antes de prosseguir.
- **Não é Fase 5:** nenhum benchmark de tempo/BD aqui. A Fase 4 só prova fidelidade + inércia.
