# Relatório técnico — Pipeline de extração de ground truth de particionamento (AV1 All-Intra)

**Contexto:** laboratório de otimização de codificador AV1; linha de pesquisa em heurísticas de
particionamento. Branch `test-partition-heuristic`.
**Objetivo:** instrumentar o libaom (v3.10.0) para extrair, durante a codificação **All-Intra**,
o *ground truth* das decisões de particionamento do RDO junto com os blocos de luminância (Y),
QP e dimensões do quadro, gravando em binário estruturado; e um conversor Python que produz
datasets `.pkl` reprodutíveis para treinar um modelo substituto (surrogate) **ConvNeXt-AV1** que
prevê o particionamento a partir do conteúdo da imagem.

Este documento registra **tudo que foi feito, decidido, encontrado e validado**. Para o passo a
passo operacional, ver [`GUIA_partition_dataset.md`](GUIA_partition_dataset.md).

---

## 1. Visão geral da arquitetura

```
                 (Docker, cpu-used=0, --threads=1)
 .yuv UVG 4K ──► aomenc (libaom + LOG_PARTITION_DATA) ──► av1_partition_data.bin
                          │                                      │
                 hook em av1_rd_pick_partition          registros fixos de 4116 B
                                                                 │
                                        convert_partition_data.py ├──► dataset .pkl (numpy)
                                        validate_partition_data.py├──► integridade + acurácia
                                                                 │       de pixel + PNGs
                                        build_dataset.py ────────┘ orquestra N seq × N QP + manifest.csv
```

Três camadas:
1. **Instrumentação C** (compilação condicional) — gera o binário durante o encode.
2. **Scripts Python** (`src/scripts/partition_dataset/`) — conversão, validação e orquestração.
3. **Manifesto/artefatos** (`results/dataset/`) — datasets `.pkl`, `.bin` e `manifest.csv`.

---

## 2. Instrumentação C

**Arquivo:** `av1/encoder/partition_search.c`. Todo o código está sob `#if LOG_PARTITION_DATA`
(default `0` — sem a flag o binário é idêntico ao de produção, sem overhead).

- **Ponto de captura:** função `av1_rd_pick_partition()` (caminho **RD**, usado em All-Intra de
  boa qualidade). É recursiva: uma chamada por nó da quadtree.
  - Os pixels Y são copiados **logo após `av1_set_offsets()`** (topo da função), antes de a
    recursão mover os ponteiros de plano.
  - A **decisão** (`pc_tree->partitioning`) é capturada quando `found_best_partition` é
    verdadeiro, **antes** de o `pc_tree` ser liberado no nível de superbloco, e o registro é
    gravado no fim da função.
- **Escopo:** apenas `cpi->oxcf.mode == ALLINTRA` e blocos **quadrados** de interesse
  (`BLOCK_64X64/32X32/16X16/8X8`).
- **Profundidade de bits:** Y sempre gravado em **8-bit**; se `is_cur_buf_hbd(xd)`, os pixels
  high-bitdepth são reduzidos (`>> (bd-8)`).
- **Saída:** arquivo apontado por `AV1_PARTITION_LOG` (fallback `av1_partition_data.bin`),
  aberto uma vez em modo append binário; `sample_id` é um contador global por processo.

Funções auxiliares adicionadas: `av1_partition_target_dim`, `av1_partition_capture_luma`,
`av1_partition_log_write`.

---

## 3. Formato binário (`PartitionSample`, 4116 bytes)

Little-endian, campos escalares ordenados do maior para o menor, buffer de pixels por último com
padding explícito → **sem padding implícito**; parsing em Python com um único `struct` format
`"<I4H5B3x4096s"`.

| Campo | Tipo | Bytes | Origem |
|-------|------|------:|--------|
| `sample_id` | u32 | 4 | contador determinístico (reinicia por processo) |
| `frame_width` | u16 | 2 | `cm->width` |
| `frame_height` | u16 | 2 | `cm->height` |
| `mi_row` | u16 | 2 | posição (unidades MI = 4 px) |
| `mi_col` | u16 | 2 | posição (unidades MI = 4 px) |
| `qindex` | u8 | 1 | `cm->quant_params.base_qindex` (0..255) |
| `bit_depth` | u8 | 1 | 8/10/12 |
| `bsize` | u8 | 1 | `BLOCK_SIZE` |
| `block_dim` | u8 | 1 | lado válido: 8/16/32/64 |
| `partition` | u8 | 1 | `PARTITION_TYPE` do RDO (0..9) |
| `pad[3]` | u8×3 | 3 | alinhamento |
| `luma[64*64]` | u8×4096 | 4096 | Y da fonte; região válida = `block_dim²`, resto zerado |

Buffer fixo de 64×64 cobre os 4 tamanhos (blocos menores usam o canto superior-esquerdo). O
rótulo é o `PARTITION_TYPE` **completo (10 classes)** — pode ser reformulado como binário
split/no-split no Python.

---

## 4. Scripts Python (`src/scripts/partition_dataset/`)

Ficam sob `src/` porque o `docker-compose` monta apenas `./src`, `./logs`, `./results` no
container — repo-root não é visível dentro do Docker.

| Script | Papel |
|--------|-------|
| `convert_partition_data.py` | `.bin` → `.pkl`. Valida alinhamento (4116), recorta a região válida, normaliza (`/255` float32 ou `uint8` cru), empilha em ndarray denso quando uniforme, grava metadados + contagens por classe/tamanho. Filtros `--block-dim`, `--qindex`. |
| `validate_partition_data.py` | Integridade + **acurácia de pixel** contra o YUV fonte + export de PNGs distribuídos por (tamanho × classe). **Frame-aware** via `--frame-offsets`: detecta limites de frame pelo reset do `sample_id` e compara cada segmento contra o frame-fonte correto. |
| `build_dataset.py` | Orquestrador N sequências × N QPs. Lê dims do nome do arquivo e nº de frames do tamanho; amostra frames espaçados no tempo; **pré-extrai** cada frame-alvo uma vez com `dd` (seek) reusando entre QPs; encoda, converte e escreve `manifest.csv`. Flags: `--qps`, `--frames`, `--cpu-used`, `--keep-bin/--no-keep-bin`, `--dry-run`. |

**Dependências:** o Python do sistema no container não tem numpy (PEP 668) → usar venv
(`python3 -m venv /tmp/venv && /tmp/venv/bin/pip install numpy pillow`).

---

## 5. Decisões metodológicas

- **`--cpu-used=0` para o ground truth.** Presets mais rápidos ligam *pruning* por ML e
  *early-termination* que **não avaliam** vários tipos de partição, empobrecendo o rótulo.
  Evidência empírica (Beauty, frame 0, cq32): a cpu-used=6 as classes `HORZ_A/B`, `VERT_A/B`,
  `HORZ_4`, `VERT_4` deram **ZERO**; a cpu-used=0 aparecem todas. Densidade também cresce muito:
  **~125.775 amostras/frame a cpu-used=0** vs ~34.593 a cpu-used=3 (~3,6×). Datasets com
  cpu-used diferente **não são comparáveis** como ground truth — o valor é registrado no manifesto.
- **Rótulo completo (10 classes)** em vez de binário — preserva informação; a redução para
  binário pode ser feita a posteriori.
- **Amostragem temporal estratificada.** Em All-Intra, frames consecutivos são quase idênticos
  (redundância temporal) → dataset pouco representativo. O `build_dataset.py` amostra **K frames
  uniformemente espaçados** no clipe inteiro (offsets `i·(total−1)/(K−1)`), maximizando
  diversidade com custo mínimo. A diversidade vem de: espacial (≈dezenas de milhares de blocos/
  frame), sequências, QPs e frames espaçados.
- **8-bit** para os pixels (foco UVG 8-bit/4:2:0; highbd reduzido).
- **Determinismo:** `--threads=1` e `--cpu-used`/QP fixos. O logging usa estado global (arquivo +
  contador), **não é thread-safe** por design — priorizou-se determinismo.

---

## 6. Achados técnicos (gotchas)

1. **`aomenc --limit` conta frames a partir do início do input, ANTES do `--skip`.** Logo
   `--skip=599 --limit=1` codifica **0 frames** (confirmado: "0 decoded frames"). Isso fez a
   amostragem temporal, na primeira geração, gravar **apenas o frame 0**. Correção: pré-extrair o
   frame com `dd skip=` e encodar com `--limit=1` (equivalente seguro: `--skip=off --limit=off+1`).
2. **Presets muito rápidos não usam o caminho RD.** A cpu-used=8 **nada** foi logado (o
   all-intra cai no caminho nonrd/var-based, sem a instrumentação). O logging depende de
   `av1_rd_pick_partition` — usar cpu-used baixo (0).
3. **Append entre processos funciona.** Cada `aomenc` reabre o `.bin` em append; encodar o mesmo
   frame 2× dobra os registros (37.256 → 74.512). O `sample_id` reinicia a cada processo — por
   isso ele **marca o limite entre frames** (usado pelo validador frame-aware).
4. **Otimização de I/O.** `--skip` relê o arquivo desde o início a cada chamada (~4 min para
   chegar ao frame 599 num volume montado a ~31 MB/s). A pré-extração com `dd` (seek direto,
   ~12 MB/frame) reutilizada entre QPs elimina ~2–3 h de I/O redundante no run completo.
5. **Desbalanceamento de classe.** ~**91,8%** das amostras são `PARTITION_NONE`; ~**69%** são
   blocos 8×8. O treino do ConvNeXt precisará de balanceamento/pesos/loss adequada e,
   possivelmente, modelagem por tamanho de bloco.

---

## 7. Validação

**Metodologia.** Três níveis:
- **Integridade estrutural:** `tamanho % 4116 == 0`, `partition ∈ [0,9]`, `block_dim ∈ {8,16,32,64}`.
- **Acurácia de pixel (crítico para ML):** para blocos totalmente internos ao quadro, os bytes
  `luma[]` gravados devem ser **idênticos** aos pixels do plano Y da **fonte** em
  `(mi_row·4, mi_col·4)`. Prova que cada bloco carrega a região certa, alinhada ao seu rótulo.
  O validador é frame-aware: separa segmentos pelo reset do `sample_id` e compara cada um contra
  o frame-fonte correto.
- **PNG:** blocos gerados **a partir do `.bin`** (não do YUV), distribuídos por tamanho × classe,
  para inspeção visual independente.

**Resultados.**
- Frame único (Beauty, cq32, cpu-used=3): 34.593 amostras, **0 mismatch** em 34.439 blocos internos.
- Dataset reduzido final (4 seq × cq32 × 2 frames `[0,599]`, cpu-used=0): **817.540 amostras**,
  **2 segmentos** detectados em todas as sequências, **0 mismatch** em **816.390** blocos internos.
- Todos os **10 tipos de partição** presentes em todas as sequências.
- Bitstream válido (`aomdec` decodifica); `qindex=128` e `3840×2160` consistentes.

---

## 8. Dataset atual gerado (`results/dataset/`)

4 sequências UVG × `--cq-level=32` (base_qindex=128) × 2 frames `[0, 599]`, cpu-used=0,
threads=1. Um `.bin` + um `.pkl` por sequência + `manifest.csv`.

| Sequência | Amostras | 8px | 16px | 32px | 64px | NONE | SPLIT | Retangulares+4* |
|-----------|---------:|----:|-----:|-----:|-----:|-----:|------:|----------------:|
| Beauty | 243.675 | 168.363 | 56.009 | 15.370 | 3.933 | 220.226 | 9.395 | 14.054 |
| HoneyBee | 211.780 | 142.185 | 51.235 | 14.500 | 3.860 | 195.587 | 7.200 | 8.993 |
| Jockey | 193.360 | 125.958 | 49.550 | 14.125 | 3.727 | 179.913 | 6.939 | 6.508 |
| Bosphorus | 168.725 | 103.022 | 46.838 | 14.902 | 3.963 | 154.998 | 4.251 | 9.476 |
| **Total** | **817.540** | 539.528 | 203.632 | 58.897 | 15.483 | 750.724 | 27.785 | — |

\* HORZ+VERT+HORZ_A/B+VERT_A/B+HORZ_4+VERT_4. Tamanho em disco: `.pkl` ≈ 833 MB, `.bin` ≈ 3,2 GB.

---

## 9. Limitações conhecidas

- **Cobertura parcial:** o dataset validado usa **apenas cq32** e **2 frames**. O run completo
  planejado é 4 QP (20/32/43/55) × 5 frames (`--frames 5`).
- **Sem índice de frame na struct:** a proveniência de frame é inferida pelo reset do `sample_id`
  (validador via `--frame-offsets`). Se for necessário rastrear frame por amostra, adicionar um
  campo exigiria mudar o layout de 4116 bytes.
- **Thread-safety:** logging só com `--threads=1`.
- **`.pkl` (pickle)** é específico de Python — adequado ao pipeline de treino, não como formato
  de intercâmbio.
- **Desbalanceamento** (item 6.5) é do domínio, não do pipeline — tratar no treino.

---

## 10. Reprodutibilidade

- **Ambiente:** container `research_env` (ver [`GUIA_builds.md`](GUIA_builds.md)); build
  instrumentada em `build/libaom_logpart` com `-DCMAKE_C_FLAGS="-DLOG_PARTITION_DATA=1"` e
  `AOM_TARGET_CPU=generic`.
- **Sequências:** UVG 4K 8-bit 4:2:0 em `src/samples/*.yuv` (Beauty, Bosphorus, HoneyBee,
  Jockey; 600 frames cada).
- **Comando canônico do dataset reduzido:**
  ```bash
  python /workspace/src/scripts/partition_dataset/build_dataset.py \
    --out-dir /workspace/results/dataset --qps 32 --frames 2 --cpu-used 0 \
    --python /tmp/venv/bin/python
  ```
- **Determinismo:** `--threads=1`, cpu-used/QP fixos, offsets de frame determinísticos.
- Todos os parâmetros de cada geração ficam registrados em `manifest.csv`.

---

## 11. Próximos passos

1. **Run completo** (4 QP × 5 frames × 4 seq, cpu-used=0, `--no-keep-bin`) — ~5–6 h estimadas.
2. **Split treino/validação por sequência** (evitar vazamento de conteúdo entre splits).
3. **Pipeline ConvNeXt-AV1**: consumir os `.pkl`, com estratégia de balanceamento de classe e/ou
   modelagem por tamanho de bloco.
