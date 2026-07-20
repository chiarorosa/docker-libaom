# C1 — Custo de busca por candidato de partição (a alavanca C3 existe?)

**Data:** 2026-07-20
**Bloco 6 (engenharia C), item C1.** Decide, **sem gastar um experimento de BD×tempo**,
se a alavanca C3 (podar só AB/4-way) tem custo a recuperar.
**Artefato:** `results/benchmark/partstats/part_timing.csv`
**Script de análise:** `src/scripts/benchmark/analyze_partstats.py`

---

## 1. Pergunta e critério de falsificação

O relatório geral conjecturava que AB (HORZ_A/B, VERT_A/B) e 4-way (HORZ_4, VERT_4)
seriam uma fatia pequena do custo do nó, o que mataria a alavanca C3. O plano fixou o
critério: **se AB+4-way somarem < 10% do tempo de busca local do nó, C3 morre sem
experimento.** C1 mede exatamente esse share.

A instrumentação **já existe no upstream** (`CONFIG_COLLECT_PARTITION_STATS`,
`aom_config_defaults.cmake:141`) — não exigiu escrever código, só um build separado.
Cada nó de `av1_rd_pick_partition` emite uma linha em `part_timing.csv` com os
microssegundos gastos em cada um dos 10 candidatos (`times[EXT_PARTITION_TYPES]`,
`partition_search.c:3344-3370`).

## 2. Método

- **Build separado instrumentado** `libaom_partstats` (`-DCONFIG_COLLECT_PARTITION_STATS=1`,
  `AOM_TARGET_CPU=generic`, RelWithDebInfo) — **não** misturado com benchmarks BD×tempo,
  porque a instrumentação altera o tempo de parede (o share, não o absoluto, é o
  entregável).
- **Encode:** Jockey 3840×2160, All-Intra, cpu-used=0, cq-level=32, 2 quadros — mesmo
  regime da tese. Comando idêntico ao `encode_ctc.py` (`--end-usage=q`, kf-dist=0,
  deltaq=0, tpl=0).
- **Amostra:** 236.763 nós de decisão (linhas de 49 campos válidas).

### Cuidado metodológico (essencial)

O timer de `PARTITION_SPLIT` **engloba a recursão** (`start` em
`partition_search.c:4552`, `end` em `:4610`, cercando o laço recursivo `:4556-4609`):
`times[SPLIT]` inclui o tempo de **todos os nós-descendentes**. Somar a coluna SPLIT
contaria o mesmo trabalho múltiplas vezes. Portanto o **tempo de trabalho local** de um
nó é a soma dos candidatos **não-recursivos** — NONE+HORZ+VERT+AB+4-way — e a coluna
SPLIT é **deliberadamente excluída** (seu custo já aparece nas linhas dos filhos). O
share reportado é a fração do trabalho de avaliação de candidatos, somado sobre todos os
nós do encode.

## 3. Resultado — C3 é JUSTIFICADA, não falsificada

Share do tempo de busca local por candidato (global, 236.763 nós):

| candidato | tempo (s) | **share** | attempts |
|---|--:|--:|--:|
| NONE | 141,3 | 28,5% | 234.818 |
| HORZ | 77,5 | 15,6% | 36.126 |
| VERT | 75,9 | 15,3% | 35.529 |
| HORZ_A | 31,0 | 6,2% | 8.117 |
| HORZ_B | 28,7 | 5,8% | 9.240 |
| VERT_A | 36,2 | 7,3% | 7.902 |
| VERT_B | 28,1 | 5,7% | 9.102 |
| HORZ_4 | 40,1 | 8,1% | 14.823 |
| VERT_4 | 37,6 | 7,6% | 14.294 |

**Agregados-chave:**

| pool | share do tempo local |
|---|--:|
| NONE | 28,5% |
| RECT (HORZ+VERT) | 30,9% |
| AB (HORZ/VERT_A/B) | 25,0% |
| 4-way (HORZ/VERT_4) | 15,7% |
| **AB + 4-way** | **40,6%** ← alavanca C3 |
| RECT+AB+4-way | 71,5% (cota superior: `av1_disable_rect_partitions` derruba os três) |

**A falsificação FALHA por larga margem: AB+4-way = 40,6% do tempo de busca local do nó,
4× acima do limiar de 10%.** É um pool de custo grande e real. A conjectura do relatório
geral ("AB/4-way são fatia pequena") está **errada** neste regime (cpu-used=0, All-Intra,
onde tudo é buscado exaustivamente).

### Onde o custo se concentra (por tamanho de bloco)

| bsize | tempo local (s) | share AB+4-way |
|---|--:|--:|
| 8×8 | 31,4 | 0,0% (AB/4-way não se aplicam) |
| 16×16 | 76,4 | 1,6% |
| 32×32 | 193,8 | **49,8%** |
| 64×64 | 109,3 | **62,0%** |
| 128×128 | 85,3 | **42,3%** |

O custo de AB+4-way vive quase inteiramente nos **blocos grandes** (≥32×32) — onde metade
ou mais do trabalho de candidatos é partição estendida. Nos blocos pequenos é
desprezível. Isto é coerente com o gating nativo por bsize (`ext_partition_eval_thresh`,
`speed_features.c:1243`).

## 4. Consequência para a tese

1. **É um pool de custo ORTOGONAL ao que o H9a já explora.** O H9a implantado age em
   NONE-commit (corta subárvores) e SPLIT-force; **não** poda AB/4-way seletivamente.
   Os ~40% presos em partição estendida são território que nenhuma solução da tese
   tocou — reforça a lacuna de caracterização que motiva o trabalho futuro do B3
   (política direcional) e abre um alvo novo e mensurável.
2. **C3 não é falsificada — mas sua realização não é grátis.** O achado C3 do plano
   permanece: **não dá para podar só AB/4-way com os helpers atuais** —
   `av1_disable_rect_partitions` derruba RECT+AB+4-way juntos
   (`encodeframe_utils.h:253-258`). Realizar C3 exige (a) esforço médio em C (expor
   `ab_partitions_allowed[]`/`part4_search_allowed[]` em `PartitionSearchState`), **ou**
   (b) a sonda barata `ext_partition_eval_thresh`, que desliga AB/4-way por bsize
   globalmente e serve de **cota superior do ganho** sem escrever código.
3. **O que C1 entrega:** a decisão. Antes de C1, C3 poderia estar morta (custo <10%).
   Agora sabemos que há ~40% de custo local em jogo, concentrado nos blocos grandes —
   então **vale medir a cota superior** (ext_partition_eval_thresh) antes de investir no
   C de esforço médio. C1 justifica dar esse passo; não o executa.

## 5. Limitações

- **Build genérico (sem SIMD):** o share pode diferir ~alguns pp de um build com SIMD,
  porque a aceleração vetorial atinge de forma desigual as etapas ricas em transformada.
  A ordem de grandeza (40,6% vs limiar 10%) é robusta a essa distorção; um número exato
  de speedup exigiria build de produção — fora do escopo de C1.
- **Share ≠ speedup de parede.** O tempo local soma ~496 s dos ~557 s de parede do
  encode instrumentado (~89%); o restante é recursão/entropia/pós-filtros. AB+4-way ≈ 40,6%
  do local ≈ ~36% da parede — mas o speedup **realizável** depende de quanto se poda com
  segurança, não do custo bruto. C1 mede o custo disponível, não o ganho.
- **1 sequência, 2 quadros.** Suficiente para a decisão (236 mil nós, share estável e
  4× acima do limiar), mas a magnitude por-bsize pode variar entre conteúdos. Não é uma
  medição de fronteira.
- **`threads=2` gerou ~6,7% de linhas rasgadas** (escrita concorrente em `fopen("a")`),
  descartadas na análise; são interleaves aleatórios, não um subconjunto sistemático.
  Uma reexecução `threads=1` (CSV sem rasgos) confirma os mesmos shares — ver §6.

## 6. Confirmação single-thread (CSV sem linhas rasgadas)

Reexecução idêntica com `threads=1` → CSV **sem rasgos** (276.425 linhas, **todas** com
49 campos; 0 descartadas). Os shares reproduzem o `threads=2` dentro de **<1 pp**:

| pool | threads=2 (236.763 nós) | **threads=1 (276.425 nós)** |
|---|--:|--:|
| NONE | 28,5% | 27,9% |
| RECT (HORZ+VERT) | 30,9% | 30,8% |
| AB | 25,0% | 25,5% |
| 4-way | 15,7% | 15,8% |
| **AB + 4-way** | **40,6%** | **41,3%** |

Concentração por bsize (threads=1): 32×32 = 50,8%, 64×64 = 62,2%, 128×128 = 41,6%,
16×16 = 1,9%, 8×8 = 0% — idêntica ao `threads=2` em padrão e magnitude. **A decisão de
C1 (C3 justificada, ~4× acima do limiar) é robusta à contagem de threads e às linhas
rasgadas.** CSV limpo: `results/benchmark/partstats/part_timing_t1.csv`.

## 7. Reprodução

```bash
# build instrumentado (uma vez)
cmake -S /workspace/src/aom -B /workspace/build/libaom_partstats -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DAOM_TARGET_CPU=generic \
  -DCONFIG_COLLECT_PARTITION_STATS=1 -DENABLE_TESTS=OFF
cmake --build /workspace/build/libaom_partstats -j"$(nproc)"

# encode curto (part_timing.csv no cwd)
cd /workspace/results/benchmark/partstats
/workspace/build/libaom_partstats/aomenc -w 3840 -h 2160 --fps=120/1 \
  --input-bit-depth=8 --cpu-used=0 --passes=1 --end-usage=q --cq-level=32 \
  --kf-min-dist=0 --kf-max-dist=0 --deltaq-mode=0 --enable-tpl-model=0 \
  --enable-keyframe-filtering=0 --threads=1 --row-mt=0 --bit-depth=8 \
  --limit=2 --psnr --obu -o out.obu \
  /workspace/src/samples/Jockey_3840x2160_120fps_420_8bit_YUV_RAW.yuv

python3 /workspace/src/scripts/benchmark/analyze_partstats.py part_timing.csv
```
