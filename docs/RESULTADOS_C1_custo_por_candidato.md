# C1/C3 — Custo de busca por candidato de partição no conjunto de teste congelado

**Data:** 2026-07-20
**Bloco 6 (engenharia C), item C1** — e base de justificativa da alavanca **C3**.
Decide, **sem gastar um experimento de BD×tempo**, se AB/4-way têm custo a recuperar,
e mede-o sobre **todo o conjunto de teste congelado (Jockey, RaceNight, RiverBank)**.
**Artefatos:** `results/benchmark/partstats*/part_timing*.csv` (não versionados, ~34 MB cada)
**Script de análise:** `src/scripts/benchmark/analyze_partstats.py`

---

## 1. Pergunta e critério de falsificação

O relatório geral conjecturava que AB (HORZ_A/B, VERT_A/B) e 4-way (HORZ_4, VERT_4)
seriam uma fatia pequena do custo do nó, o que mataria a alavanca C3. O plano fixou o
critério: **se AB+4-way somarem < 10% do tempo de busca local do nó, C3 morre sem
experimento.** C1 mede esse share.

A instrumentação **já existe no upstream** (`CONFIG_COLLECT_PARTITION_STATS`,
`aom_config_defaults.cmake:141`) — não exigiu escrever código, só um build separado.
Cada nó de `av1_rd_pick_partition` emite uma linha em `part_timing.csv` com os
microssegundos gastos em cada um dos 10 candidatos (`times[EXT_PARTITION_TYPES]`,
`partition_search.c:3344-3370`).

## 2. Método

- **Build separado instrumentado** `libaom_partstats` (`-DCONFIG_COLLECT_PARTITION_STATS=1`,
  `AOM_TARGET_CPU=generic`, RelWithDebInfo) — **não** misturado com benchmarks BD×tempo,
  porque a instrumentação altera o tempo de parede (o share, não o absoluto, é o entregável).
- **Encodes:** os **três** de teste congelados — Jockey, RaceNight, RiverBank — 3840×2160,
  All-Intra, cpu-used=0, cq-level=32, 2 quadros, `threads=1` (CSV **sem linhas rasgadas**).
  Comando idêntico ao `encode_ctc.py`.
- **Amostra:** **875.317 nós** de decisão (276k + 318k + 281k), zero linhas descartadas.

### Cuidado metodológico (essencial)

O timer de `PARTITION_SPLIT` **engloba a recursão** (`start` em `partition_search.c:4552`,
`end` em `:4610`, cercando o laço recursivo `:4556-4609`): `times[SPLIT]` inclui o tempo de
**todos os nós-descendentes**. Somar a coluna SPLIT contaria o mesmo trabalho múltiplas
vezes. Portanto o **tempo de trabalho local** de um nó é a soma dos candidatos
**não-recursivos** — NONE+HORZ+VERT+AB+4-way — e a coluna SPLIT é **deliberadamente
excluída** (seu custo já aparece nas linhas dos filhos). O share reportado é a fração do
trabalho de avaliação de candidatos, somado sobre todos os nós.

## 3. Resultado — C3 é JUSTIFICADA em todo o conjunto de teste

### Por sequência (share do tempo de busca local)

| sequência | nós | NONE | RECT | AB | 4-way | **AB+4-way** |
|---|--:|--:|--:|--:|--:|--:|
| Jockey | 276.425 | 27,9% | 30,8% | 25,5% | 15,8% | **41,3%** |
| RaceNight | 317.708 | 29,7% | 37,9% | 18,0% | 14,4% | **32,5%** |
| RiverBank | 281.184 | 32,9% | 38,2% | 17,6% | 11,3% | **28,9%** |
| **agregado** | **875.317** | **30,1%** | **35,6%** | **20,4%** | **13,9%** | **34,3%** |

**AB+4-way varia de 28,9% a 41,3%, com agregado de 34,3% — o mínimo já é ~3× o limiar de
falsificação (10%).** A conjectura do relatório geral ("AB/4-way são fatia pequena") está
**errada** neste regime (cpu-used=0, All-Intra, onde tudo é buscado exaustivamente), e o
resultado é **consistente entre conteúdos**, não um artefato de uma sequência.

### Decomposição agregada por candidato (875.317 nós)

| candidato | tempo (s) | share | attempts |
|---|--:|--:|--:|
| NONE | 526,2 | 30,10% | 871.180 |
| HORZ | 314,2 | 17,98% | 180.095 |
| VERT | 308,6 | 17,66% | 178.173 |
| HORZ_A | 93,0 | 5,32% | 36.135 |
| HORZ_B | 82,4 | 4,71% | 37.322 |
| VERT_A | 100,9 | 5,77% | 30.698 |
| VERT_B | 79,5 | 4,55% | 31.869 |
| HORZ_4 | 126,2 | 7,22% | 48.671 |
| VERT_4 | 117,0 | 6,69% | 45.593 |

| pool | share do tempo local agregado |
|---|--:|
| NONE | 30,1% |
| RECT (HORZ+VERT) | 35,6% |
| AB | 20,4% |
| 4-way | 13,9% |
| **AB + 4-way** | **34,3%** ← alavanca C3 |
| RECT+AB+4-way | 69,9% (cota superior: `av1_disable_rect_partitions` derruba os três) |

### Onde o custo se concentra (por tamanho de bloco, agregado)

| bsize | tempo local (s) | share AB+4-way |
|---|--:|--:|
| 8×8 | 136,9 | 0,0% (AB/4-way não se aplicam) |
| 16×16 | 416,7 | 8,7% |
| 32×32 | 657,5 | **50,0%** |
| 64×64 | 284,1 | **51,0%** |
| 128×128 | 252,7 | **35,3%** |

O custo de AB+4-way vive quase inteiramente nos **blocos grandes** (≥32×32) — metade do
trabalho de candidatos a partir de 32×32. Desprezível em ≤16×16. Coerente com o gating
nativo por bsize (`ext_partition_eval_thresh`, `speed_features.c:1243`).

**Veredito:** AB+4-way = 34,3% agregado (mín. 28,9%), ~3–4× acima do limiar de 10% em
**todas** as três sequências → **C3 não é falsificada; é justificada.**

## 4. Consequência para a tese — substrato para um novo modelo H9

1. **É um pool de custo ORTOGONAL ao que o H9a já explora.** O H9a implantado age no eixo
   primário (NONE-commit / SPLIT-force); **não** poda AB/4-way seletivamente. Os ~34% presos
   em partição estendida são um **terceiro eixo de decisão** que nenhuma solução da tese
   tocou — território novo, grande e mensurável.
2. **C3 não é falsificada — mas sua realização não é grátis.** O achado C3 do plano permanece:
   não dá para podar só AB/4-way com os helpers atuais — `av1_disable_rect_partitions` derruba
   RECT+AB+4-way juntos (`encodeframe_utils.h:253-258`). Realizar C3 exige (a) esforço médio em
   C (expor `ab_partitions_allowed[]`/`part4_search_allowed[]` em `PartitionSearchState`), **ou**
   (b) a sonda barata `ext_partition_eval_thresh`, que desliga AB/4-way por bsize globalmente e
   serve de **cota superior do ganho** sem escrever código.

### 4.1 Proposta de nomenclatura (eixos de decisão da família H9)

A família H9 organiza-se por **eixo de decisão de partição**, e o C1 mostra que o eixo
estendido é grande o bastante para sustentar um modelo próprio — como o H9c teve nome
distinto por atacar um refinamento do NONE:

| modelo | eixo | ação | estado |
|---|---|---|---|
| **H9a** | primário | NONE-commit / SPLIT-force | implantado |
| **H9c** | refinamento do NONE (39 feats, +`none_rdcost`) | pós-NONE | medido e rejeitado (real) |
| **H9d** | **estendido (AB/4-way)** | podar a busca de partição estendida — o pool de **34% do tempo local** | **medido e rejeitado** — teto dominado pelo H9a (`RESULTADOS_H9d_cota_superior.md`) |
| **H9e / H9-rect** (proposto) | orientação retangular | desligar só a direção errada (B3, 69% direcional) | trabalho futuro (`RESULTADOS_modelagem_B3_horz_vert.md §6`) |

**H9d** é o casamento direto do achado C1 com a alavanca C3: um preditor que decide, por nó
(nos blocos grandes onde o custo vive), se vale buscar AB/4-way. É onde mora o ganho de tempo
não-explorado. **H9e/H9-rect** (o B3) é um eixo relacionado mas distinto — orientação, não
"buscar ou não o estendido"; fica como trabalho futuro separado.

> **Nota de escopo:** este C1 mostra que o pool AB/4-way existe e é grande e consistente — mas
> **não** basta para justificar o H9d. A cota superior foi medida em seguida
> (`RESULTADOS_H9d_cota_superior.md`) e o veredito **inverteu**: o teto do H9d (blanket AB/4-way
> off, 1,43×/0,89% BD) é **dominado nos dois eixos pelo H9a já implantado** nas mesmas 3 seqs
> (cpu0), porque o NONE-commit do H9a já pula AB/4-way ao cortar a subárvore inteira. **H9d fica
> como lever medido-e-rejeitado** — o custo existe, mas o modelo em produção já o colhe melhor.

## 5. Limitações

- **Build genérico (sem SIMD):** o share pode diferir ~alguns pp de um build de produção,
  porque a aceleração vetorial atinge de forma desigual as etapas ricas em transformada. A
  ordem de grandeza (28,9–41,3% vs limiar 10%) é robusta a essa distorção; um número exato de
  speedup exigiria build de produção — fora do escopo de C1.
- **Share ≠ speedup de parede.** O tempo local agregado (1748 s) é ~89% da parede dos encodes
  instrumentados; o restante é recursão/entropia/pós-filtros. AB+4-way ≈ 34% do local; o
  speedup **realizável** depende de quanto se poda com segurança, não do custo bruto. C1 mede
  o custo disponível, não o ganho.
- **2 quadros por sequência, um único CQ (cq32).** Suficiente para a decisão (875 mil nós,
  share estável e ~3–4× acima do limiar nas três sequências), mas a magnitude por-bsize pode
  variar com CQ e conteúdo. Não é medição de fronteira.
- **CSV single-thread (`threads=1`):** zero linhas rasgadas nas três sequências (49 campos em
  todas as linhas). Um teste anterior com `threads=2` (Jockey) descartou ~6,7% de linhas
  rasgadas por escrita concorrente e reproduziu o mesmo share dentro de <1 pp — confirmando
  que a decisão é robusta à contagem de threads.

## 6. Reprodução

```bash
# build instrumentado (uma vez)
cmake -S /workspace/src/aom -B /workspace/build/libaom_partstats -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DAOM_TARGET_CPU=generic \
  -DCONFIG_COLLECT_PARTITION_STATS=1 -DENABLE_TESTS=OFF
cmake --build /workspace/build/libaom_partstats -j"$(nproc)"

# por sequência (part_timing.csv no cwd) — repetir para RaceNight/RiverBank
cd /workspace/results/benchmark/partstats
/workspace/build/libaom_partstats/aomenc -w 3840 -h 2160 --fps=120/1 \
  --input-bit-depth=8 --cpu-used=0 --passes=1 --end-usage=q --cq-level=32 \
  --kf-min-dist=0 --kf-max-dist=0 --deltaq-mode=0 --enable-tpl-model=0 \
  --enable-keyframe-filtering=0 --threads=1 --row-mt=0 --bit-depth=8 \
  --limit=2 --psnr --obu -o out.obu \
  /workspace/src/samples/Jockey_3840x2160_120fps_420_8bit_YUV_RAW.yuv

# agregado dos três de teste
python3 /workspace/src/scripts/benchmark/analyze_partstats.py \
  partstats/part_timing_t1.csv partstats_racenight/part_timing.csv \
  partstats_riverbank/part_timing.csv --label=Jockey,RaceNight,RiverBank
```
