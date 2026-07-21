# H9d — Etapa 2: integração da política seletiva em C

**Data:** 2026-07-21
**H9d, Etapa 2 (cirurgia no codificador).** Implementa o podador seletivo de partição
estendida (AB/4-way) como estágio **pós-NONE**, no molde do H9c. Segue a Etapa 1
(`RESULTADOS_H9d_predizibilidade.md`, AUC 0,90) e a cota superior
(`RESULTADOS_H9d_cota_superior.md`). O modelo de 39 features (h9c-set) é gravado no C.

---

## 1. O que foi feito

### 1.1 Pesos → header C (com round-trip verificado)
`src/scripts/partition_model/h9d_export_weights.py` exporta o MLP de 39 features (2 saídas
NAO_EXT/EXT, por nível 16/32/64) para `av1/encoder/partition_student_h9d_weights.h`, no
layout que `av1_nn_predict` espera. **Verificação de round-trip:** para 192 vetores
aleatórios, `|softmax(PyTorch) − softmax(layout C)| = 1,35e-07` (exato até ponto flutuante)
— garante que os pesos gravados no codificador são os treinados.

### 1.2 Decisão em C (espelha o H9c)
Em `partition_strategy.c` (sob `#if PARTITION_ML_STUDENT`):
- `student_h9d_enabled()` — lê `AV1_STUDENT_H9D_ENABLE` (**off por padrão**).
- `student_h9d_get_tau(n)` — lê `AV1_STUDENT_H9D_TAU[_16/_32/_64]` (default 0,10).
- `student_h9d_decide()` — computa as **mesmas 39 features do H9c** (`student_node_features`
  + `log1p(none_rate/dist/rdcost)`), roda `av1_nn_predict`, softmax, e **marca o nó para
  pular AB/4-way quando `P(EXT) < τ`** (`part_state->h9d_skip_ext = 1`). NONE/rect/split
  ficam intocados.
- Ligado em `av1_prune_after_none` (`partition_search.c:5841`, pós-NONE, antes do gate
  AB em `:5903`), com a mesma condição de gate do H9c (intra, sb≥64, bsize∈{16,32,64},
  unidade 64×64 inteira no quadro). H9c e H9d rodam **independentes** (envs separadas).

### 1.3 Consumo do flag nos gates AB/4-way
`PartitionSearchState` ganhou `int h9d_skip_ext` (init 0 em
`init_partition_search_state_params`). Nos gates AB (`partition_search.c:~4108`) e 4-way
(`~4174`), a condição existente do probe blanket foi estendida:
`if (ext_off || part_search_state->h9d_skip_ext) thresh = BLOCK_128X128;` — desliga AB/4-way
naquele nó. Reusa o mecanismo já validado do gate `AV1_EXT_PART_OFF`.

## 2. Superfície de controle (tudo por env-var, off por padrão)

| env | efeito | default |
|---|---|---|
| `AV1_STUDENT_H9D_ENABLE` | liga o podador seletivo H9d | off (inerte) |
| `AV1_STUDENT_H9D_TAU` | limiar global: pula AB/4-way se `P(EXT) < τ` | 0,10 |
| `AV1_STUDENT_H9D_TAU_16/_32/_64` | limiar por nível (sobrepõe o global) | = global |

Sem a env, o código é **no-op** (o braço "nativo"/H9a é byte-idêntico). Mesma filosofia
opt-in do H9a/H9c — nenhum eixo contamina o outro. Requer build com
`-DPARTITION_ML_STUDENT=1` (como o H9a/H9c).

## 3. Validação de sanidade (Jockey cq32, 1 quadro)

| config | tempo | PSNR-Y |
|---|--:|--:|
| H9a sozinho (H9d off) | 18,4s | 39,8340 |
| H9a + H9d (τ=0,10) | 17,8s | 39,8310 |
| H9a + H9d (τ=0,30) | 15,8s | 39,8290 |

- **Inércia confirmada:** H9d off reproduz o H9a puro (18,4s / 39,8340), idêntico ao binário
  pré-H9d → o código é inerte sem a env.
- **Pruner dispara e é monótono:** ligar o H9d acelera (pula AB/4-way); τ maior → mais
  agressivo → mais rápido, com queda mínima de PSNR. Comportamento correto.

## 4. Paridade de features (herdada do H9c, já validada)

O `student_h9d_decide` usa **exatamente** o mesmo `student_node_features` + contexto RD
pós-NONE do `student_h9c_decide` — o vetor de 39 features cuja paridade C↔Python já está
estabelecida e implantada no H9c (`features.py::node_features_h9c`: "The C side ... mirrors
THIS"). Logo a paridade do H9d é herdada, não precisa de nova verificação.

## 5. Próximo passo — Etapa 3 (confirmação no encoder)

Falta o árbitro final: sweep de encodes ≥10 quadros, 3 seqs de teste, comparando
**H9a+H9d_seletivo** (varrendo τ) contra **H9a sozinho** e a **curva de τ do H9a**. Se o H9d
seletivo dominar o blanket e a curva de τ (como a Etapa 1 projeta), fecha a contribuição.

## 6. Reprodução

```bash
# 1. exportar pesos (round-trip check embutido)
/workspace/build/venv-ml/bin/python \
  src/scripts/partition_model/h9d_export_weights.py
# 2. build com o estudante compilado
cmake --build /workspace/build/libaom_extoff_ml -j"$(nproc)"
# 3. encode com H9a (taus P_ref) + H9d
AV1_STUDENT_TAU_NONE_16=0.85 ... AV1_STUDENT_H9D_ENABLE=1 AV1_STUDENT_H9D_TAU=0.30 \
  aomenc --cpu-used=0 ... (ver src/scripts/benchmark/h9d_upper_bound.py)
```
