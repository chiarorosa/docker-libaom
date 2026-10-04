# Custo implantado de cada podador do NPL-AV1 (pré-busca, pós-NONE e estendidas)

Data: 2026-10-04. Objeto: estender a medição de `docs/RESULTADOS_microbench_pruner.md` §6, que cobria
só o podador pré-busca (H9a, ≤0,32% do tempo de codificação), aos podadores pós-NONE (H9c) e de
partições estendidas (H9d), para que o artigo ISCAS reporte o custo de **todo** o método (norma B.9
do grupo).

## 1. Instrumentação

`src/aom/av1/encoder/partition_strategy.c`, bloco `AV1_PRUNER_TIMING` (opt-in em tempo de execução,
inerte quando desligado):

- três acumuladores novos: `g_pt_h9c_feat`, `g_pt_h9d_feat`, `g_pt_h9d_inf`;
- a extração de atributos dos podadores pós-NONE e de estendidas passou a ser cronometrada (antes,
  só a inferência do pós-NONE era);
- **defeito corrigido**: a inferência do H9d era somada em `g_pt_h9c_inf` (cópia de código).

Build separado, para não alterar o binário da campanha D2: `build/libaom_perf_timing` (Release,
`-DPARTITION_ML_STUDENT=1`, `CONFIG_INTERNAL_STATS=0`, `ENABLE_TESTS=0`), sem aviso de compilação no
arquivo alterado.

## 2. QA

1. **Inércia**: BoxingPractice cq32, `cpu-used=1`, 3 quadros, pré-busca efficiency-first + H9d, CNN
   desligada: MD5 idêntico (`609ea381dfee91e12f915175712366e1`) entre `libaom_perf_h9d`,
   `libaom_perf_timing` com timing desligado e com timing ligado.
2. **Concorrência**: com `--threads=2` (dois *tiles*), as contagens de extração e de inferência
   divergiram por 1–2 chamadas — os acumuladores são estáticos e não sincronizados. A medição usa
   `--threads=1`, como o protocolo original (§6.2); nela as contagens coincidem em todas as linhas.
3. **Reprodutibilidade**: chamadas da CNN 5.940 (Tango cq32) e 5.938 (BoxingPractice cq43), contra
   5.940 e 5.937 em julho.

## 3. Resultados (3 quadros, `cpu-used=1`, `--threads=1`; % do tempo de parede da mesma codificação)

| sequência | configuração | parede (s) | CNN | pré-busca | pós-NONE | estendidas |
|---|---|--:|--:|--:|--:|--:|
| Tango cq32 | CNN nativa | 112,7 | 0,133% | (0,245%)ᵃ | — | — |
| Tango cq32 | pós-NONE substituto | 106,5 | — | (0,270%)ᵃ | **0,258%** | — |
| Tango cq32 | efficiency-first + estendidas | 89,1 | — | **0,408%** | — | **0,368%** |
| BoxingPractice cq43 | CNN nativa | 78,7 | 0,187% | (0,315%)ᵃ | — | — |
| BoxingPractice cq43 | pós-NONE substituto | 79,0 | — | (0,376%)ᵃ | **0,361%** | — |
| BoxingPractice cq43 | efficiency-first + estendidas | 70,2 | — | **0,482%** | — | **0,421%** |

ᵃ Pré-busca **neutralizado** (limiares 2/2/−1): extrai e infere, mas nunca decide. É custo do arranjo
de medição das campanhas de substituição, não do podador que está sendo avaliado.

Artefato: `results/benchmark/overhead_iscas/overhead.csv`.

## 4. Leitura

1. Cada podador custa **menos de meio por cento** do tempo de codificação: pós-NONE 0,26–0,36%,
   pré-busca efficiency-first 0,41–0,48%, estendidas 0,37–0,42%, contra 0,13–0,19% da CNN nativa.
   A configuração mais cara, pré-busca + estendidas, soma **0,78–0,90%**.
2. As percentagens do pré-busca sobem em relação a julho (0,26–0,32%) porque, no ponto
   efficiency-first, ele é consultado em mais nós (poda menos cedo) e a codificação é mais curta.
3. O podador de estendidas **recalcula** os 36 atributos que o pré-busca já extraiu no mesmo nó
   (`student_node_features` chamada de novo): a extração é cerca de 3/4 do seu custo. Reaproveitar o
   vetor do pré-busca reduziria o custo combinado; não foi implementado.
4. Nas campanhas de substituição do pós-NONE, o pré-busca neutralizado custou 0,27–0,38% sem podar
   nada; os TS publicados do pós-NONE são, portanto, ligeiramente **conservadores**.

## 5. Limitações

1. Duas sequências, um cq cada, 3 quadros e `cpu-used=1`, como o protocolo de julho; não varre
   presets nem cq.
2. `--threads=1` é condição da medição, e não a configuração das campanhas (`--threads=2`, CTC 4K).
3. O tempo de `clock_gettime` (dezenas de ns por chamada) é somado ao do podador.

## 6. Reprodução (contêiner `av1_bench`)

```bash
cmake -G Ninja -S src/aom -B build/libaom_perf_timing -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_FLAGS=-DPARTITION_ML_STUDENT=1 -DCONFIG_INTERNAL_STATS=0 -DENABLE_TESTS=0 -DENABLE_DOCS=0
ninja -C build/libaom_perf_timing aomenc
bash src/scripts/fase6/qa_timing_inertness.sh                      # QA de inércia (MD5)
build/venv-ml/bin/python src/scripts/fase6/overhead_pruners_iscas.py
```
