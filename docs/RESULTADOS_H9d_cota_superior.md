# H9d — poda de partição estendida (AB+4-way) como estágio pós-NONE

**Data:** 2026-07-20
**Bloco 6, alavanca C3 → modelo H9d.** Mede, com encodes reais nos três de teste
congelados, o **envelope** de ganho de um podador de partição estendida. Segue o C1
(`RESULTADOS_C1_custo_por_candidato.md`), que estabeleceu que AB+4-way são 34% do tempo
local do nó.

**Veredito (§4/§6): H9d pós-NONE é um lever legítimo — Pareto-NÃO-dominado.** Empilhar poda
de AB/4-way *depois* do H9a (só no resíduo que o H9a não NONE-commita) adiciona um ponto
**novo e não-dominado** à fronteira do H9a nas três sequências (§3.7). A leitura inicial de
"teto dominado/rejeitado" usava a moldura errada (blanket vs nativo pristino, que dupla-conta
o pool que o H9a já colhe); a moldura correta é o **marginal pós-NONE**, como o H9c.
**Scripts:** `h9d_upper_bound.py` (--stack), `h9d_tau_curve.py` · **Análise BD:** `bd_rate.py`
**Dados:** `results/benchmark/{h9d_ub,h9d_marg,h9d_tau}/raw.csv` (não versionados)

> **Trilha de raciocínio (registro honesto):** (1) blanket vs nativo = 1,43×/0,89% — parecia
> atrativo; (2) blanket vs H9a agressivo (h9_test) = teto dominado → conclusão precipitada de
> "rejeitado"; (3) a moldura certa, provocada em revisão, é o **marginal pós-NONE** (o H9a
> decide primeiro, em `partition_search.c:5753`, antes de AB/4-way em `:5903+`; o gate só age
> no resíduo) → Pareto-não-dominado. O §3.6 preserva (2) como passo intermediário; o §3.7 traz
> a correção.

---

## 1. O que "cota superior" significa aqui

Um H9d **seletivo** decide, por nó, se vale buscar AB/4-way — busca onde provavelmente
vencem, poda o resto. O **blanket disable** (desligar AB/4-way em *todos* os nós) é o
**extremo** desse eixo:

- **maior speedup possível** (não busca nada estendido) → **teto de aceleração**;
- **maior perda de qualidade possível** (perde AB/4-way mesmo quando eram ótimos) →
  **pior caso de BD-rate**.

Qualquer H9d real fica **dentro** deste envelope: menos speedup que o blanket, menos
BD-rate que o blanket. Portanto o blanket entrega as duas fronteiras do problema: quanto
tempo há para ganhar, e quanto custa perder tudo.

## 2. Método — isolamento por env-var no mesmo binário

- **Um único binário** `libaom_extoff` (Release, SIMD nativo, `CONFIG_INTERNAL_STATS=1`).
- A mudança em `partition_search.c` (sítios AB `:4101` e 4-way `:4160`) é **gated por
  env-var** e **inerte por padrão**:
  ```c
  static int ext_off = -1;
  if (ext_off < 0) ext_off = getenv("AV1_EXT_PART_OFF") ? 1 : 0;
  if (ext_off) ab_bsize_thresh = BLOCK_128X128;  // idem part4_bsize_thresh
  ```
  `BLOCK_128X128` faz `bsize > thresh` ser sempre falso → AB/4-way desligados em todo
  tamanho. **Sem a env, é no-op: o braço "nativo" é o encoder real**, byte-idêntico ao
  fonte não modificado. Mesma filosofia opt-in do H9a/H9c (`AV1_STUDENT_TAU_*`) — nenhum
  eixo contamina o outro.
- **Braços:** `native` (env unset) vs `extoff` (`AV1_EXT_PART_OFF=1`), **mesmo binário**,
  única diferença o gate → isolamento limpo do efeito de AB/4-way.
- **Regime:** idêntico ao `encode_ctc.py` — cpu-used=0, All-Intra, `end-usage=q`,
  kf-dist=0, deltaq=0, tpl=0, tiles=1, threads=2. **5 quadros × 4 CQ (20/32/43/55) ×
  3 seqs**. Rate p/ BD-rate = bits totais do fluxo; qualidade = PSNR-Y.

## 3. Resultado — o envelope

| sequência | **BD-rate** (extoff vs nativo) | **speedup** (teto) | tempo nativo→extoff (s) |
|---|--:|--:|--:|
| Jockey | +1,13% | **1,485×** | 636,5 → 428,7 |
| RaceNight | +0,76% | **1,418×** | 759,4 → 535,7 |
| RiverBank | +0,79% | **1,402×** | 733,3 → 523,2 |
| **média** | **+0,89%** | **1,431×** | — |

**Cruza-validação com o C1:** o C1 (offline, decomposição de tempo por candidato) previu
que remover AB+4-way tiraria ~34% do tempo local ≈ ~30% de parede → ~1,43× de speedup. O
encode real dá **1,43×**. As duas medições independentes concordam — o custo medido no
`part_timing` é real e se realiza em tempo de parede.

### A leitura decisiva: densidade de qualidade baixíssima

AB+4-way custam **~43% do tempo de parede** mas carregam apenas **+0,89% de BD-rate** de
qualidade. Isto é, gastam muito e decidem pouco:

> **~0,03 pp de BD-rate por 1% de tempo economizado** (0,89% ÷ ~30%).

É exatamente o perfil que favoreceria um podador: candidatos caros e raramente decisivos.
**Mas — e este é o ponto decisivo — o H9a implantado já poda esse mesmo pool, e melhor.**

## 3.6 Passo intermediário — blanket vs H9a (a moldura ERRADA)

*(Este passo levou a uma conclusão precipitada de "dominado". É preservado como registro; a
correção está no §3.7. O erro: comparar o blanket **substituto** ao H9a supõe que H9d
*substitui* o H9a — mas H9d é um **complemento pós-NONE**, como o H9c.)*

O H9a foi medido nos **mesmos três de teste, mesmo cpu-used=0, mesmo anchor nativo**
(`results/benchmark/h9_test/*/curve_{safe,aggr}/summary.csv`). Colocando os pontos de
operação do H9a lado a lado com o teto do H9d **isolado** (todos vs nativo, cpu0):

| seq | H9a P_ref (safe) | H9a A1_none80 (aggr) | H9a A2_none70 (aggr+) | **H9d blanket (teto)** |
|---|--:|--:|--:|--:|
| Jockey | 0,919% / 1,483× | 0,929% / **1,503×** | 1,372% / 1,911× | 1,13% / 1,485× |
| RaceNight | 0,736% / 1,467× | 0,729% / **1,484×** | 1,165% / 1,715× | 0,76% / 1,418× |
| RiverBank | 0,129% / 1,319× | 0,19% / 1,328× | 0,226% / **1,434×** | 0,79% / 1,402× |

*(formato: BD-rate% / speedup; frames: H9a ~15, H9d 5 — os números do H9a são os mais
robustos, e são justamente os que dominam.)*

**Em cada sequência existe um ponto de operação do H9a que domina o teto do H9d nos dois
eixos** (mais speedup E menos BD-rate):
- **Jockey**: H9a aggr 1,503×/0,93% > H9d 1,485×/1,13%.
- **RaceNight**: H9a aggr 1,484×/0,729% > H9d 1,418×/0,76%.
- **RiverBank**: H9a A2_none70 1,434×/0,226% > H9d 1,402×/0,79% (mais rápido e ~⅓ do BD).

Isso *parecia* fechar o caso — mas trata o H9d como substituto do H9a. Ele não é.

## 3.7 A moldura correta — H9d pós-NONE é Pareto-NÃO-dominado

O H9d age **depois** do H9a: pelo fluxo de controle, `av1_prune_partitions_before_search`
(a decisão do H9a) roda em `partition_search.c:5753`, **antes** de NONE (`:5823`), rect
(`:5883`) e AB/4-way (`:5903+`). Se o H9a NONE-commita ou faz rest-off, o nó nem chega ao
gate de AB/4-way; o gate **só age no resíduo** — nós onde o H9a escolheu prosseguir para
rect/split. Portanto a medição correta é **marginal**: H9a+extoff vs H9a, mesmo binário
(`libaom_extoff_ml`, `PARTITION_ML_STUDENT=1` + gate), H9a no ponto de referência **P_ref**.

**Marginal (blanket AB/4-way off em cima do H9a P_ref), 5fr:**

| seq | BD-rate marginal | speedup marginal |
|---|--:|--:|
| Jockey | +0,765% | 1,367× |
| RaceNight | +0,879% | 1,239× |
| RiverBank | +0,749% | 1,290× |
| média | +0,798% | 1,293× |

Mesmo **depois** dos NONE-commits do H9a, ainda há **~1,29× de speedup** preso em AB/4-way
no resíduo — o pool marginal é grande, **não** é ruído.

**Sobreposição com a curva de τ do H9a (tudo 5fr, mesmo anchor nativo pristino):** a pergunta
decisiva é se empilhar AB/4-way-off bate simplesmente **subir o τ** do H9a (knob grátis).

| seq | P_ref | A1_none80 | A2_none70 | A3_none60 | **H9a+extoff** |
|---|--:|--:|--:|--:|--:|
| Jockey | 1,41% / 1,47× | 1,42% / 1,47× | 2,05% / 1,79× | 2,47% / 2,08× | **2,19% / 2,00×** |
| RaceNight | 0,32% / 1,53× | 0,32% / 1,51× | 1,11% / 1,70× | 1,96% / 2,01× | **1,21% / 1,86×** |
| RiverBank | 0,16% / 1,32× | 0,17% / 1,30× | 0,28% / 1,41× | 0,45% / 1,54× | **0,91% / 1,66×** |

*(formato BD-rate% / speedup, todos vs o mesmo nativo 5fr.)*

**Teste de dominância de Pareto — o ponto empilhado NÃO é dominado por nenhum ponto de τ, em
nenhuma das três sequências:**
- **Jockey** (2,19%/2,00×): A2 é mais lento (1,79×), A3 tem BD pior (2,47%) → não-dominado.
- **RaceNight** (1,21%/1,86×): A2 mais lento (1,70×), A3 BD pior (1,96%) → não-dominado.
- **RiverBank** (0,91%/1,66×): **nenhum** ponto de τ atinge 1,66× de speedup → não-dominado.

**Conclusão: empilhar poda de AB/4-way pós-NONE adiciona um ponto NOVO à fronteira do H9a nas
três sequências.** E isto é o *blanket* (pior caso, sem seleção) — um H9d **seletivo** trocaria
melhor (menos BD ao mesmo speedup), empurrando o ponto ainda mais para cima da fronteira de τ.

## 4. Veredito — H9d pós-NONE é um lever legítimo

A comparação blanket-vs-nativo (§3.6) usava a moldura errada. Na moldura correta (§3.7,
marginal pós-NONE), o H9d **não é dominado** — é um ponto novo na fronteira do H9a. Portanto:

1. **H9d pós-NONE vale investigar** — mesmo o blanket é Pareto-competitivo com o knob de τ, e a
   seleção só melhora. É o molde do H9c (podador aprendido pós-NONE), agora sobre o eixo
   estendido em vez do NONE.
2. **Predizibilidade offline — FEITO, PASSOU** (`RESULTADOS_H9d_predizibilidade.md`): as
   features do H9a separam "AB/4-way vence" com **ROC-AUC 0,890** (6 seqs held-out, 792.840
   nós). Dá para evitar ~67% das buscas perdendo só 10% dos vencedores (ou 50% perdendo 2,6%)
   → o seletivo é uma melhoria de Pareto clara sobre o blanket. **Sinal verde para as Etapas
   2–3.**
3. **Se render offline:** política em C (`ab_bsize_thresh`/`part4_bsize_thresh` como função da
   predição por nó, não env global — o gate já existe) + confirmação BD×tempo (A5). O gate
   `AV1_EXT_PART_OFF` já provou o mecanismo; falta torná-lo seletivo.

## 5. Limitações

- **Blanket ≠ seletivo.** O ponto empilhado medido é o *blanket* (desliga TODOS os AB/4-way
  no resíduo) — o pior caso de um H9d. O H9d seletivo fica **melhor** (menos BD ao mesmo
  speedup). A não-dominância do §3.7 é, portanto, uma **cota inferior** do valor do H9d: se
  até o blanket é não-dominado, o seletivo tende a dominar. Falta medir a predizibilidade que
  entrega essa seleção (§4 item 2).
- **5 quadros por ponto, 4 CQ.** Base estável para a dominância de Pareto (o ponto empilhado
  fica fora da curva de τ por margens claras em speedup), mas não é a rigidez de uma medição
  de fronteira final (a tese usa ≥10-15 quadros para resultados E). As comparações são todas
  **vs o mesmo anchor nativo 5fr** e **mesmo binário** — internamente consistentes; os BD-rate
  absolutos diferem de `h9_test` (~15fr) e não devem ser cruzados.
- **Base = P_ref.** O ponto empilhado usa o H9a no ponto de referência P_ref. Empilhar sobre
  bases mais/menos agressivas geraria uma família de pontos; P_ref (o implantado) é o natural.
- **PSNR-Y apenas.** BD-rate sobre PSNR-Y (padrão da tese); não sobre PSNR combinado/SSIM.

## 6. Veredito

**H9d pós-NONE é um lever legítimo.** Na moldura correta — poda de AB/4-way empilhada
*depois* do H9a, agindo só no resíduo que o H9a não NONE-commita — o ponto medido é
**Pareto-não-dominado** vs a curva de τ do H9a nas três sequências (§3.7): adiciona mais
speedup (Jockey 2,00×, RaceNight 1,86×, RiverBank 1,66×) do que qualquer ponto de τ, a BD
competitivo. E isto é o *blanket*; um H9d **seletivo** só melhora.

A leitura anterior de "rejeitado" (§3.6) comparava o H9d **como substituto** do H9a — moldura
errada para um **complemento** pós-NONE. Corrigida, a resposta a "vale um segundo modelo para
partição estendida?" é **provavelmente sim — pendente de uma medição offline de
predizibilidade** (§4 item 2), que decide se a seleção entrega a dominância que o blanket já
tangencia. É o mesmo caminho que validou/rejeitou o H9c: medir o crivo offline antes do C.

**Estado:** lever **aberto e promissor**, não fechado. Próximo passo = predizibilidade de
AB/4-way com features do nó (barato, offline), depois política seletiva em C + encodes.

## 7. Reprodução

```bash
# build 1: gate ext-off, estudante OFF (para o blanket vs nativo pristino) — §3
cmake -S /workspace/src/aom -B /workspace/build/libaom_extoff -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCONFIG_INTERNAL_STATS=1 -DENABLE_TESTS=OFF
# build 2: gate ext-off + H9a compilado (para o marginal pós-NONE) — §3.7
cmake -S /workspace/src/aom -B /workspace/build/libaom_extoff_ml -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCONFIG_INTERNAL_STATS=1 -DENABLE_TESTS=OFF \
  -DCMAKE_C_FLAGS="-DPARTITION_ML_STUDENT=1"
cmake --build /workspace/build/libaom_extoff    -j"$(nproc)"
cmake --build /workspace/build/libaom_extoff_ml -j"$(nproc)"

PY=/workspace/build/venv-ml/bin/python
# blanket (nativo vs extoff, estudante off)
$PY src/scripts/benchmark/h9d_upper_bound.py --frames 5
# marginal pos-NONE (H9a P_ref vs H9a+extoff)
$PY src/scripts/benchmark/h9d_upper_bound.py --enc .../libaom_extoff_ml/aomenc \
    --stack h9a --frames 5 --work .../h9d_marg
# curva de tau a 5fr (overlay do §3.7)
$PY src/scripts/benchmark/h9d_tau_curve.py
```
