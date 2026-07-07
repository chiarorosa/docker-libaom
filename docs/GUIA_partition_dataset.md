# GUIA — Geração de dataset de particionamento (ground truth do RDO)

Guia operacional para extrair, da codificação **All-Intra** do libaom, o *ground truth* das
decisões de particionamento (para treinar o surrogate **ConvNeXt-AV1**) e converter em datasets
`.pkl`. Para o registro técnico completo (decisões, achados, validação), ver
[`RELATORIO_pipeline_dataset_particionamento.md`](RELATORIO_pipeline_dataset_particionamento.md).

## Componentes

| Camada | Onde | Papel |
|--------|------|-------|
| Instrumentação C | `av1/encoder/partition_search.c` (`#if LOG_PARTITION_DATA`) | grava `.bin` durante o encode |
| `build_dataset.py` | `src/scripts/partition_dataset/` | **orquestra** N seq × N QP → `.pkl` + `manifest.csv` |
| `convert_partition_data.py` | idem | `.bin` → `.pkl` (usado pelo orquestrador ou avulso) |
| `validate_partition_data.py` | idem | integridade + acurácia de pixel + PNGs |

Os scripts ficam sob `src/` porque o container monta só `./src`, `./logs`, `./results`.
Sem a flag `LOG_PARTITION_DATA=1`, o encoder é idêntico ao de produção (nenhuma amostra).

## Formato do registro (`PartitionSample`, 4116 bytes, little-endian)

`sample_id`(u32), `frame_width`/`frame_height`/`mi_row`/`mi_col`(u16), `qindex`/`bit_depth`/
`bsize`/`block_dim`/`partition`(u8), `pad[3]`, `luma[64*64]`(u8). Parsing Python:
`struct.Struct("<I4H5B3x4096s")`. Rótulo = `PARTITION_TYPE` completo (0..9). Pixels em 8-bit;
região válida = `block_dim²` no canto superior-esquerdo do buffer 64×64.

---

## 1. Build instrumentada (uma vez, no container)

Diretório de build **separado** para não contaminar o loop diário ([`GUIA_builds.md`](GUIA_builds.md)):

```bash
cmake -S /workspace/src/aom -B /workspace/build/libaom_logpart \
  -G Ninja -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -DAOM_TARGET_CPU:STRING=generic \
  -DENABLE_CCACHE:BOOL=1 -DENABLE_EXAMPLES:BOOL=ON -DENABLE_TESTS:BOOL=OFF \
  -DENABLE_DOCS:BOOL=OFF -DCONFIG_INTERNAL_STATS:BOOL=1 \
  -DCMAKE_C_FLAGS="-DLOG_PARTITION_DATA=1"

cmake --build /workspace/build/libaom_logpart -j"$(nproc)"
```

## 2. Ambiente Python (numpy não vem no container — PEP 668)

```bash
python3 -m venv /tmp/venv
/tmp/venv/bin/pip install numpy pillow      # pillow só p/ os PNGs do validador
```

## 3. Gerar o dataset (caminho recomendado: `build_dataset.py`)

Varre `src/samples/*.yuv`, amostra frames **espaçados no tempo**, encoda em All-Intra
(`--usage=2`, `--threads=1`), converte para `.pkl` e escreve `manifest.csv`.

```bash
/tmp/venv/bin/python /workspace/src/scripts/partition_dataset/build_dataset.py \
  --out-dir /workspace/results/dataset \
  --qps 20 32 43 55 \        # cq-level (0..63); base_qindex é derivado (cq32 -> 128)
  --frames 5 \               # K frames uniformemente espaçados no clipe
  --cpu-used 0 \             # 0 = RD quase exaustivo -> ground truth fiel (ver relatório)
  --python /tmp/venv/bin/python \
  --no-keep-bin              # apaga o .bin após converter (economiza disco)
```

Flags úteis: `--dry-run` (mostra o plano e sai), `--seqs <arquivo.yuv ...>` (sequências
explícitas), `--samples-dir`, `--keep-bin` (padrão, mantém o `.bin`).

**Saída:** `results/dataset/<seq>_cq<qp>.pkl` (+`.bin` se mantido) e `manifest.csv` com dims,
frames usados, `cq_level`, `base_qindex`, `cpu_used`, nº de amostras e contagem por tamanho/classe.

> **Importante (cpu-used):** use **0** para o ground truth. Presets rápidos podam classes
> inteiras de partição (a cpu-used≥8 o all-intra nem passa pelo caminho instrumentado). Ver
> relatório, seção "Achados".

## 4. Validar uma extração (`validate_partition_data.py`)

Confere integridade, **acurácia de pixel contra o YUV fonte** e exporta PNGs (de tamanhos/classes
variados) a partir do `.bin`. Requer o `.bin` (rode antes do `--no-keep-bin`, ou sem ele).
Para `.bin` com múltiplos frames acumulados, passe os offsets em `--frame-offsets`:

```bash
/tmp/venv/bin/python /workspace/src/scripts/partition_dataset/validate_partition_data.py \
  --bin    /workspace/results/dataset/Beauty_..._cq32.bin \
  --source /workspace/src/samples/Beauty_..._8bit_YUV_RAW.yuv \
  --width 3840 --height 2160 \
  --frame-offsets 0 599 \                        # mesma ordem usada na geração
  --out-dir /workspace/results/dataset/blocks/Beauty --num-png 2
```

Saída esperada: `frame segments detected: N`, `pixel-accuracy: ... 0 mismatched`,
`RESULT: PASS`.

## 5. Política de retenção (`.bin` × `.pkl`)

Depois que a validação passa, **só o `.pkl` é obrigatório** — ele é a entrada de treino do
surrogate. O `.bin` é opcional e serve apenas a etapas *anteriores* ao ML:

| Arquivo | Papel | Guardar para o ML? |
|---------|-------|--------------------|
| `.pkl` | dataset convertido (entrada de treino) | **Sim** — sempre |
| `manifest.csv` | metadados (`cpu_used`, `base_qindex`, frames, contagem por classe) | **Sim, sempre** — sem ele o `.pkl` perde rastreabilidade e datasets com cpu-used diferente não são comparáveis |
| `.bin` | registros brutos `PartitionSample` (4116 B) | **Não** — o treino nunca lê o `.bin` |

O `.bin` só é necessário para: (1) **validar** (`validate_partition_data.py` lê o `.bin`) e
(2) **reconverter** o `.pkl` com outras opções (`--dtype`, `--block-dim`, `--qindex`, filtros)
sem re-encodar.

**Regra prática:**
- Se o `.pkl` já está no dtype/filtros definitivos do treino → descarte o `.bin` (`--no-keep-bin`).
  Ficam só `.pkl` + `manifest.csv`.
- Se ainda pode haver reconversão → **mantenha o `.bin`** (`--keep-bin`, padrão). A alternativa
  é re-encodar, e a cpu-used=0 custa ~510 s por frame 4K — o `.bin` é um cache barato contra um
  recompute caríssimo.

Ou seja: manter o `.bin` é uma decisão de **disco × custo de recompute**, não uma exigência do ML.

## 6. Conversão avulsa (opcional, `convert_partition_data.py`)

Se já tiver um `.bin` (ex.: gerado manualmente com o `aomenc` e `AV1_PARTITION_LOG`):

```bash
/tmp/venv/bin/python /workspace/src/scripts/partition_dataset/convert_partition_data.py \
  --input  /workspace/results/av1_partition_data.bin \
  --output /workspace/results/dataset_uvg_cq32.pkl \
  --seq minha_sequencia
# filtros: --block-dim 32   --qindex 128   --dtype float32|uint8
```

Encode manual de 1 frame no offset `K` (cuidado com o gotcha do `--limit`):

```bash
export AV1_PARTITION_LOG=/workspace/results/av1_partition_data.bin; rm -f "$AV1_PARTITION_LOG"
/workspace/build/libaom_logpart/aomenc --usage=2 --passes=1 --threads=1 --cpu-used=0 \
  --end-usage=q --cq-level=32 -w 3840 -h 2160 --skip=K --limit=$((K+1)) \
  -o /workspace/results/out.ivf src/samples/<seq>.yuv
```
> `--limit` conta frames do início do input, **antes** do `--skip` — por isso `--limit=K+1`
> (não `--limit=1`). O `build_dataset.py` já contorna isso pré-extraindo o frame com `dd`.

## 7. Checklist de verificação

- Build **sem** `-DLOG_PARTITION_DATA=1` não gera `.bin`.
- `tamanho_do_bin % 4116 == 0`; validador lê tudo sem sobra; `partition ∈ [0,9]`.
- Acurácia de pixel: `0 mismatched` nos blocos internos.
- Determinismo: `--threads=1` + cpu-used/QP fixos → `.bin` reprodutível.
- `frame segments detected` == nº de frames amostrados.
