# Zenodo — datasheet e texto de registro do dataset

Documento para publicar o dataset com DOI. Duas partes: (1) o **texto do
registro** (cola no campo *Description* do Zenodo) e (2) o **datasheet técnico**
(esquema + loader de referência), para acompanhar o depósito.

> **Formato recomendado:** `.npz` uint8 (gerado por
> `src/scripts/partition_dataset/pkl_to_npz.py`), ~8–12 GB, portável e sem
> pickle. Cabe nos 50 GB / 100 arquivos do Zenodo (64 `.npz` + manifest +
> datasheet). Alternativa: os `.pkl` float32 (31 GB) — funcionais, porém frágeis
> (pickle dependente de versão); se usar, documente o ambiente exato.

---

## Parte 1 — Texto de registro (Description)

**Título:** AV1 All-Intra Partition Decisions with Rate-Distortion Context — 4K
Ground-Truth Dataset (libaom v3.10.0, cpu-used=0)

**Descrição:**
Ground truth das decisões de particionamento do codificador de referência AV1
(libaom v3.10.0) em modo All-Intra, extraído com busca RD completa (`cpu-used=0`),
o único regime em que todas as classes de partição são exploradas. Para cada nó
de particionamento quadrado (8/16/32/64 px) de 16 sequências UVG 4K, registra-se
a luminância do bloco-fonte, o **rótulo de partição** escolhido pelo RD (uma de 10
classes PARTITION_TYPE) e o **contexto de taxa-distorção** que a decisão usa:
contexto de particionamento dos vizinhos (tamanhos dos blocos acima/esquerda),
força de quantização (passo de dequantização DC), e o custo RD real do
PARTITION_NONE (taxa, distorção, rdcost). O dataset foi construído para estudar
heurísticas de poda de particionamento aprendidas (aceleração do codificador com
perda de taxa BD desprezível). ~27,5 milhões de nós, 16 sequências × 4 QPs
(cq 20/32/43/55) × 5 quadros amostrados temporalmente. Partição
treino/validação/teste **por sequência** (sem vazamento) fixada a priori.

**Palavras-chave:** AV1, libaom, video coding, block partitioning, rate-distortion,
machine learning, encoder acceleration, All-Intra, 4K, UVG.

**Métodos:** codificador libaom v3.10.0 instrumentado (`LOG_PARTITION_DATA`);
`aomenc --usage=2 --passes=1 --cpu-used=0 --end-usage=q --cq-level={20,32,43,55}`;
5 quadros por sequência com amostragem temporal uniforme; um registro por nó
visitado na busca RD (`av1_rd_pick_partition`). Determinístico e reproduzível a
partir do código e das sequências-fonte (UVG, públicas). Base_qindex = 4·cq.

**Licença:** BSD-2-Clause + Alliance for Open Media Patent License 1.0 (herdada
do libaom, para o código-gerador). Dados derivados de sequências UVG (checar a
licença UVG para redistribuição da luminância). *Confirmar antes de publicar.*

**Como citar / reproduzir:** o código-gerador, o protocolo congelado e a
rastreabilidade completa estão no repositório
`github.com/chiarorosa/docker-libaom` (branch `ml-partition-dev`): ver
`docs/RASTREABILIDADE.md`, `docs/PROTOCOLO_avaliacao.md`,
`docs/METODOLOGIA_pipeline_ML.md`.

---

## Parte 2 — Datasheet técnico

### Arquivos
- `<Sequencia>_..._cq{20,32,43,55}.npz` — 64 arquivos, um por (sequência, QP).
- `manifest.csv` — ledger autoritativo: por (sequência, QP), quadros usados,
  base_qindex, cpu-used, nº de amostras, histograma por tamanho e por classe.
- `ZENODO_datasheet.md` — este documento.

### Partição (congelada, sem vazamento — `PROTOCOLO_avaliacao.md`)
- **Teste:** Jockey, RaceNight, RiverBank.
- **Validação:** HoneyBee, FlowerPan, Lips.
- **Treino:** Beauty, Bosphorus, CityAlley, FlowerFocus, FlowerKids, ReadySetGo,
  ShakeNDry, SunBath, Twilight, YachtRide.

### Esquema de cada `.npz`
| Chave | dtype | forma | significado |
|---|---|---|---|
| `luma_flat` | uint8 | [ΣPᵢ] | blocos concatenados (row-major), Pᵢ = dimᵢ² |
| `luma_offsets` | int64 | [N+1] | luma da amostra i = `luma_flat[off[i]:off[i+1]]` → `(dimᵢ, dimᵢ)` |
| `partition` | uint8 | [N] | rótulo PARTITION_TYPE 0..9 (NONE, HORZ, VERT, SPLIT, HORZ_A/B, VERT_A/B, HORZ_4, VERT_4) |
| `block_dim` | uint8 | [N] | lado do bloco em px (8/16/32/64) |
| `qindex` | uint8 | [N] | base_qindex (0..255) |
| `frame_width`,`frame_height` | uint16 | [N] | dimensões do quadro |
| `mi_row`,`mi_col` | uint16 | [N] | posição em unidades mode-info (4 px) |
| `sample_id` | uint32 | [N] | contador por processo; `==0` marca fronteira de quadro |
| `above_bsize`,`left_bsize` | uint8 | [N] | **[B]** BLOCK_SIZE dos vizinhos (0 se ausente) |
| `neigh_avail` | uint8 | [N] | **[B]** bit0=tem-acima, bit1=tem-esquerda |
| `dc_q` | uint16 | [N] | **[C]** passo de dequantização DC da luma |
| `none_rate` | uint32 | [N] | **[E]** taxa do PARTITION_NONE (0 se não avaliado) |
| `none_dist` | int64 | [N] | **[E]** distorção do PARTITION_NONE |
| `none_rdcost` | int64 | [N] | **[E]** custo RD do PARTITION_NONE |
| `meta_json` | uint8 | [·] | metadados (bytes UTF-8 de um JSON) |

Notas: a luma é 8-bit exata (o `.pkl` de origem guarda `float32 = uint8/255`, sem
perda). O custo RD do NONE (`none_*`) é registrado **após** a avaliação do NONE e
serve a estudos de teto; não é usado por podadores que decidem *antes* da busca.

### Loader de referência (numpy puro, sem pickle)
```python
import numpy as np, json

def load_npz(path):
    z = np.load(path, allow_pickle=False)
    off = z["luma_offsets"]
    def luma(i):
        d = int(z["block_dim"][i])
        return z["luma_flat"][off[i]:off[i+1]].reshape(d, d)  # uint8
    meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
    return z, luma, meta

z, luma, meta = load_npz("Jockey_..._cq32.npz")
print(meta["sequence"], "N =", len(z["partition"]))
blk0 = luma(0)                      # (block_dim, block_dim) uint8
label0 = int(z["partition"][0])    # PARTITION_TYPE
none_rd0 = int(z["none_rdcost"][0])
```

### Conversão (não executada por padrão)
```bash
venv-ml/bin/python src/scripts/partition_dataset/pkl_to_npz.py \
    --in-dir results/dataset_h9 --out-dir results/dataset_h9_npz
```
Reversível conceitualmente: o `.pkl` de origem é regenerável pelo codificador
instrumentado + `convert_partition_data.py`; o `.bin` bruto idem. Backup bruto
(bin+pkl) mantido fora do Zenodo (ex.: Google Drive).
