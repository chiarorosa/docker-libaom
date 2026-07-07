# Geração de dataset de particionamento (ground truth do RDO)

Este guia descreve como extrair, da codificação **All-Intra** do libaom, os dados reais de
decisão de particionamento (ground truth do RDO) para treinar o surrogate **ConvNeXt-AV1**.

O pipeline tem duas partes:

1. **Instrumentação C** (`av1/encoder/partition_search.c`) — compilada só quando
   `LOG_PARTITION_DATA=1`. Grava um `av1_partition_data.bin` durante o encode.
2. **Conversor Python** (`src/scripts/partition_dataset/convert_partition_data.py`) — lê o `.bin`
   e produz um dataset `.pkl` normalizado. Fica sob `src/` (montado no container em
   `/workspace/src`) para ficar acessível dentro do Docker.

Sem a flag, o binário do encoder é idêntico ao de produção (nenhum overhead, nenhuma amostra).

## O que é gravado

Para cada bloco **quadrado** de interesse (`64x64`, `32x32`, `16x16`, `8x8`) avaliado pelo RD
partition search em modo All-Intra, grava-se um registro de tamanho fixo (`PartitionSample`,
**4116 bytes**, little-endian, sem padding implícito):

| Campo          | Tipo   | Origem                                    |
|----------------|--------|-------------------------------------------|
| `sample_id`    | u32    | contador global determinístico            |
| `frame_width`  | u16    | `cm->width`                               |
| `frame_height` | u16    | `cm->height`                              |
| `mi_row`       | u16    | posição do bloco (unidades MI)            |
| `mi_col`       | u16    | posição do bloco (unidades MI)            |
| `qindex`       | u8     | `cm->quant_params.base_qindex` (0..255)   |
| `bit_depth`    | u8     | 8/10/12                                   |
| `bsize`        | u8     | `BLOCK_SIZE`                              |
| `block_dim`    | u8     | lado válido em pixels: 8/16/32/64         |
| `partition`    | u8     | `PARTITION_TYPE` do RDO (0..9)            |
| `pad[3]`       | u8×3   | alinhamento                               |
| `luma[64*64]`  | u8×4096| Y da fonte; região válida = `block_dim²`  |

Pixels são sempre gravados em **8-bit** (highbd é reduzido). O rótulo é o `PARTITION_TYPE`
completo (10 classes); o Python pode reformular como binário split/no-split se necessário.

## 1. Build com a flag (dentro do container)

Use um diretório de build **separado** para não contaminar o loop diário do
[`GUIA_builds.md`](GUIA_builds.md):

```bash
cmake -S /workspace/src/aom -B /workspace/build/libaom_logpart \
  -G Ninja -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -DAOM_TARGET_CPU:STRING=generic \
  -DENABLE_CCACHE:BOOL=1 -DENABLE_EXAMPLES:BOOL=ON -DENABLE_TESTS:BOOL=OFF \
  -DENABLE_DOCS:BOOL=OFF -DCONFIG_INTERNAL_STATS:BOOL=1 \
  -DCMAKE_C_FLAGS="-DLOG_PARTITION_DATA=1"

cmake --build /workspace/build/libaom_logpart -j"$(nproc)" \
  2>&1 | tee /workspace/logs/build-libaom_logpart.log
```

## 2. Codificar uma sequência UVG (8-bit / 4:2:0)

Defina o caminho de saída via `AV1_PARTITION_LOG` (fallback: `av1_partition_data.bin` no cwd).
Use `--threads=1` e um `--cpu-used` fixo para logging **determinístico**. O modo All-Intra é
`--usage=2` (`AOM_USAGE_ALL_INTRA`); o rate control por qualidade é `--end-usage=q` com
`--cq-level` no intervalo **0..63**:

```bash
export AV1_PARTITION_LOG=/workspace/results/av1_partition_data.bin
rm -f "$AV1_PARTITION_LOG"

/workspace/build/libaom_logpart/aomenc \
  --usage=2 --passes=1 --threads=1 --cpu-used=3 --end-usage=q --cq-level=32 \
  -w <W> -h <H> --fps=30/1 --limit=<N> \
  -o /workspace/results/out.ivf <sequencia>.y4m
```

O campo `qindex` do dataset é o `base_qindex` (0..255) efetivamente usado, derivado do
`--cq-level` (ex.: `--cq-level=32` → `qindex=128`), não o valor de cq-level em si.

## 3. Converter para dataset `.pkl`

> **numpy:** o container `research_env` é de build/teste do encoder e **não traz numpy** no
> Python do sistema (PEP 668). Rode o conversor num venv (`python3 -m venv .venv &&
> .venv/bin/pip install numpy`) ou no seu ambiente de ML. O script vive em `src/scripts/`
> (montado no container em `/workspace/src/scripts`) e lê o `.bin` de `results/`.

```bash
# dentro do container (com numpy disponível no venv ativo):
python /workspace/src/scripts/partition_dataset/convert_partition_data.py \
  --input  /workspace/results/av1_partition_data.bin \
  --output /workspace/results/dataset_uvg_cq32.pkl \
  --seq minha_sequencia
# filtros opcionais: --block-dim 32   --qindex 128   --dtype float32|uint8
```

Gerar múltiplos datasets (treino/validação) é só repetir os passos 2–3 com sequências/QPs
diferentes e `--output` distintos.

## 4. Verificação

- **Compilação condicional:** build sem `-DLOG_PARTITION_DATA=1` não gera `.bin`.
- **Bitstream válido:** o encode com logging ativo ainda produz um `.ivf` decodificável
  (`aomdec`), como no teste rápido do `CLAUDE.md`.
- **Integridade:** `tamanho_do_bin % 4116 == 0`; o conversor lê tudo sem sobra de bytes;
  `partition ∈ [0,9]`; `qindex`, `frame_width/height` batem com o encode.
- **Determinismo:** duas execuções idênticas (`--threads=1`, `--cpu-used` e QP fixos) produzem
  `.bin` byte-a-byte iguais.

> Nota: o logging usa estado global (arquivo e contador). Rode com `--threads=1` — não é
> thread-safe por design, priorizando determinismo para geração de dataset.
