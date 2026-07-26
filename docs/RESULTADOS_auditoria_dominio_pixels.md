# Auditoria do domínio de pixels — o que o H9a realmente é, e o que nunca foi testado

**Data:** 2026-07-26
**Origem.** Dois documentos colocados em `docs/LUMA/` ("Arquiteturas CNN Leves — DOCIE (1) e
(2)") foram submetidos à avaliação: *cabem na tese? há ainda, no domínio de pixels, proposta
válida — implementável no codificador e competidora do H9?* A resposta aos documentos é
**não** (§5). Mas a auditoria feita para respondê-los produziu três achados sobre a própria
tese (§2–§4), um dos quais reabre uma via nunca testada (§6).

**Natureza deste documento.** Auditoria de código e de registro. Não gera medição nova de
codificação; confronta afirmações já publicadas nos documentos da tese contra a
implementação em `src/scripts/partition_model/`.

---

## 1. Método

Leitura direta da fonte, sem intermediários: `features.py` (layout do vetor),
`PLANO_H9_contribuicao_tese.md` (especificação dos blocos), checkpoints dos dois ConvNeXt
(argumentos de treino), `surrogate_replay.py` e `h7h8_bench.py` (caminho de medição do
teto H8). Cada afirmação abaixo cita arquivo e linha.

---

## 2. Achado 1 — o H9a é, majoritariamente, um modelo de pixels

O layout do vetor está rotulado no próprio código (`features.py:196-201`):

```
A  0..23  pixels
B 24..31  neighbor partition context (free)
C 32..35  quantization / position (free)
D 36..37  intra-residual proxy: Hadamard SATD of the block (cheap)
E 38..40  PARTITION_NONE rate/dist/rdcost (ceiling; not deployable pre-search)
```

Com `H9a = A+B+C` (`features.py:204`), segue que **24 dos 36 atributos do H9a são
descritores de luma** — variância global e por quadrante, gradientes h/v, orientação,
densidade de bordas, contraste com pai e irmãos (`FEATURE_NAMES`, `features.py:53-61`). E
que **o H9a não contém nenhuma grandeza de custo taxa-distorção**: `none_rate`,
`none_dist` e `none_rdcost` são o bloco E, exclusivo do H9c, que foi rejeitado.

Mais: `pixels24` é definido como `list(range(24))` (`features.py:216`) — é **literalmente o
bloco A do H9a**. Os dois braços comparados no Gate 2 e no crivo A5 não são conjuntos
disjuntos; um contém o outro.

### 2.1 A hierarquia, relida

Retornos marginais no crivo A5, em `cost_red` casado de 25% (fonte:
`RESULTADOS_convnext_regret.md §2`):

| passo | `reg_frac` | ganho marginal |
|---|--:|--:|
| `variance` (1 descritor de luma) | 0,0573 | — |
| `pixels24` (+23 descritores de luma) | 0,0121 | **4,7×** |
| `convnext_ce` (+28,1 M parâmetros sobre pixels crus) | 0,0207 | **0,6× (pior)** |
| `H9a` (`pixels24` + 12 de vizinhança/quant/posição) | 0,0036 | **3,4×** |

A leitura correta **não** é "contexto RD vence pixels". É:

> No domínio de pixels, **descritores manuais compactos vencem uma rede convolucional
> profunda sobre pixels crus**; e o que separa o campeão do resto não é capacidade de
> modelo, é **contexto causal de vizinhança** — doze atributos de custo praticamente nulo.

Isso é consistente com o achado independente de que **capacidade não é o gargalo**:
`fusion_dim` 256 contra 128 muda `val_wloss` em 0,16% (`RESULTADOS_convnext_regret.md §1.1`).

### 2.2 Por que isso importa para a banca

A afirmação antiga ("o podador implantado usa contexto RD, não pixels") é falsa e frágil:
qualquer avaliador que abrisse `features.py` a derrubaria. A afirmação nova é verdadeira,
mais forte e mais interessante — ela transforma um resultado negativo sobre CNNs num
resultado positivo sobre **onde** a informação está: na vizinhança causal, não na
capacidade de representação sobre a fonte.

---

## 3. Achado 2 — o bloco D implementado não é o bloco D especificado

`PLANO_H9_contribuicao_tese.md:326` especifica o bloco D como:

| idx | nome | fórmula |
|--:|---|---|
| 36 | `satd_pred` | SATD do **resíduo de uma predição intra barata a partir dos vizinhos reconstruídos** (PAETH ou melhor-de-{DC,PAETH,2 direcionais}) |
| 37 | `satd_gain` | `SATD(fonte) − SATD(resíduo)`, normalizado |

E adverte explicitamente (`:330-334`):

> *"Detalhe crítico (evita repetir a variância): um resíduo de predição **constante** (DC)
> tem a mesma variância do bloco — não agrega. Por isso o proxy usa SATD sobre uma predição
> **não-constante** (direcional/PAETH). Isto mede predizibilidade, que a variância não
> captura."*

A implementação (`features.py:252-260, 305-308`) calcula `block_satd(block)` — o Hadamard
do **bloco-fonte**, sem predição e sem tocar em vizinho algum — e `f[37] = satd /
(var·dim² + 1)`, razão entre duas estatísticas da própria fonte. Não existe predição intra
em lugar nenhum de `src/scripts/partition_model/` (verificado por busca de
`paeth|recon|satd_pred|satd_gain`).

**Consequência.** O "H9b" reprovado no Gate 2 (46,9 / 50,5 / 53,4 / 57,8 contra 47,0 / 49,7
/ 52,6 / 57,3 do H9a — `RASTREABILIDADE.md §5.3`) testou uma estatística **só da fonte**,
fortemente correlacionada com a variância e os gradientes que o bloco A já contém. Seu
resultado nulo era esperado e **não informa sobre a hipótese que o plano formulou**. A
advertência do próprio plano foi violada pela implementação.

**A hipótese especificada nunca foi testada.** É a lacuna que o §6 fecha.

---

## 4. Achado 3 — o custo de inferência do ConvNeXt nunca foi pago

`surrogate_replay.py:4-8` é explícito:

> *"The surrogate's input is source luma + qindex only, so its in-loop decisions can be
> precomputed EXACTLY outside the encoder. The PARTITION_ML_STUDENT hook reads this file
> back (`AV1_STUDENT_PROBS_FILE`) and applies the very same pruning policy, which turns the
> encoder into a measurement device for the surrogate's BD-rate ceiling — **no convolutional
> inference in C**."*

Portanto **todo resultado de CNN de pixels na tese (H8 e derivados) é um limite superior que
ignora o custo de inferência**. Isso é metodologicamente correto para medir um teto — e está
declarado como tal — mas precisa ser dito explicitamente nos Resultados: 28,1 M de
parâmetros por superbloco não são implantáveis na libaom em regime algum. Hoje o ponto é
inócuo (a qualidade já não passa), mas é exatamente o tipo de limitação que uma banca
aponta.

---

## 5. Avaliação dos dois documentos de `docs/LUMA/`

Ambos são levantamentos gerados por LLM sobre **backbones convolucionais leves para
classificação de imagem em tons de cinza no *edge***. O (1) recomenda MobileNetV4 (bloco
UIB/ExtraDW) e MobileOne (reparametrização estrutural); o (2) recomenda MobileNetV3-Small →
EfficientNet-Lite0 → ShuffleNetV2, com listas de datasets (MNIST, MedMNIST, ChestX-ray14,
KTH-TIPS, MVTec, UFPR-ALPR) e checklists de quantização/deployment.

**Não cabem, por três razões estruturais:**

1. **Otimizam o eixo errado.** Otimizam latência por acurácia em SoC móvel. O ramo de
   pixels da tese falhou por **qualidade**, não por custo: o `convnext_ce` perde para o H9a
   por 3–6× e para o `pixels24` por ~1,7×. Um backbone mais leve torna um modelo perdedor
   mais barato. E a capacidade já foi medida como não limitante (§2.1).
2. **Formato de tarefa incompatível.** São classificadores de imagem inteira com *global
   average pooling* → um rótulo. A tese precisa de predição densa e hierárquica: 21 nós,
   3 escalas, 10 classes, com máscara de legalidade. A recomendação se reduziria a trocar
   uma linha em `model.py:34`.
3. **O conselho central já está satisfeito.** O documento (2) recomenda *stem* de 1 canal,
   evitar descasamento de pré-treino RGB→cinza, e *split* sem vazamento. A tese já usa
   entrada de 2 canais (luma + qindex), **treina do zero** (`"pretrained": false` nos dois
   checkpoints, verificado) e separa por sequência. O protocolo de métricas que eles
   propõem (accuracy/macro-F1) seria um **retrocesso**: `RESULTADOS_approachB.md §6` já
   estabeleceu que acurácia por nó é mau proxy, e foi por isso que se construiu o crivo de
   *regret*.

**Valor residual, real mas limitado:** servem como evidência documental de que as
alternativas de backbone leve foram consideradas — útil numa nota de defesa metodológica —
e foram o que motivou a auditoria que expôs os achados §2–§4.

---

## 6. O portão D' — predizibilidade intra a partir dos vizinhos

**Hipótese (a do plano, enfim testada).** Todo atributo de pixel já medido na tese — bloco
A, `pixels24`, variância, ConvNeXt — lê **exclusivamente o bloco-fonte**. Nenhum lê a
**borda a partir da qual o codificador vai de fato extrapolar**. A evidência da própria tese
aponta para esse eixo: o bloco B (forma de partição dos vizinhos) comprou 3,4× sobre
`pixels24` (§2.1). O bloco D' é o mesmo eixo — vizinhança — com os *pixels* do vizinho em
vez da sua *forma*.

**Atributos** (`features_intrapred.py`), 3 colunas: `pred_avail`, `log_satd_resid`,
`satd_gain`. Escolhe a melhor entre {DC, V, H, PAETH} por SAD — os quatro modos que o plano
especificou — e mede a **fração de energia AC** que a predição a partir dos vizinhos remove.
PAETH é transliterado de `aom_dsp/intrapred.c` e verificado contra implementação escalar de
referência.

**Convenção do SATD.** O termo DC é descartado nas duas pontas, para que
`satd_gain = SATD(fonte) − SATD(resíduo)` compare AC com AC. Sem isso o ganho seria dominado
pelo offset médio que qualquer predição remove trivialmente — exatamente o efeito "predição
constante" contra o qual o plano advertiu. Verificado empiricamente: bloco de ruído i.i.d.
(onde DC é a melhor predição) dá `satd_gain` **exatamente 0**.

**Aproximação que torna o portão barato.** Os `.pkl` guardam a luma-**fonte** do superbloco
de 64×64, não os vizinhos reconstruídos. Usa-se a luma-fonte das linhas/colunas adjacentes
*dentro* do superbloco. Duas tendências de sinal oposto, ambas registradas:

- **Otimista:** a fonte é mais limpa que a reconstrução (sem ruído de quantização) → o
  atributo aqui é um **limite superior** da sua informação real.
- **Conservadora:** só nós com `r>0 E c>0` têm ambos os vizinhos dentro do superbloco. No
  codificador real, um nó na borda do superbloco tem vizinhos do superbloco adjacente. Em
  **64px a disponibilidade offline é zero** (r=c=0 sempre) → este portão **nada diz sobre o
  nível de 64**.

Direção líquida do viés: desconhecida. Por isso o portão é um **crivo**, não um veredito —
mesma função que o crivo A5 declara para si.

**Bancada.** Idêntica à do Gate 2 (`gate2_signal.py`): mesmo dataset, mesmo split por
sequência, mesmo MLP por nível, mesma política NONE-commit + rect-off, mesma grade de τ,
mesmos limites de risco casado. A única diferença entre os braços é o conjunto de colunas —
o veredito é sobre **informação**, não sobre capacidade.

### 6.1 Resultado

*(a preencher quando a execução terminar)*

---

## 7. Correções de registro aplicadas

| documento | afirmação anterior | correção |
|---|---|---|
| `RESULTADOS_convnext_regret.md §5` | "o podador implantado (H9a) usa contexto de taxa-distorção, **não pixels**" | falsa; substituída pela hierarquia de retornos marginais (§2.1) |
| `ANDAMENTO_tese.md §3` | rótulo de tabela "H9a (contexto RD grátis)" oposto a "pixels24" | conjuntos não são disjuntos; relabelado para "H9a (= pixels24 + vizinhança/quant/posição)" |
| `RASTREABILIDADE.md §5.3` | "contexto RD grátis supera pixels ~50 % relativo" | ganho é **marginal, não competitivo**; redação corrigida |
| `SINTESE_resultados_metodologia.md §2.5` | bloco D descrito como "proxy de resíduo intra" | é o SATD do bloco-fonte; divergência especificação↔implementação registrada |

Adicionalmente, fixou-se em `SINTESE §2.5` a **convenção de nomenclatura** de "contexto RD"
(= blocos B e C, o contexto de decisão barato que a busca nativa consulta; **não** grandezas
de custo RD, que só existem no bloco E). O termo é vocabulário consolidado da tese — inclusive
no nome da hipótese H9 — e por isso foi **definido**, não renomeado.

---

## 8. Limitações desta auditoria

- **Não mede codificação.** Nenhum encode foi executado; os achados §2–§4 são de código e
  de registro, e o §6 é offline.
- **O §6 não substitui o codificador.** Mesmo um resultado positivo exigiria reinstrumentar
  a extração para ler vizinhos **reconstruídos** e depois medir BD×tempo — o crivo elimina
  candidatos, não coroa vencedores.
- **O conjunto de preditores de D' é uma escolha de projeto.** {DC, V, H, PAETH} é o que o
  plano especificou; modos SMOOTH melhorariam conteúdo suave e não foram incluídos. Um
  resultado marginalmente negativo não excluiria um conjunto mais rico.
- **A hierarquia de §2.1 herda as ressalvas do crivo A5** — em particular a amostragem
  grosseira da curva da variância (`RESULTADOS_convnext_regret.md §3`).

---

## 9. Reprodução

```bash
# Portão D' (offline, sem codificação)
docker exec av1_bench /workspace/build/venv-ml/bin/python \
    /workspace/src/scripts/partition_model/gate_intra_pred.py \
    --out-csv /workspace/results/models/gate_intra_pred.csv

# Composição do vetor (achado 1)
sed -n '196,220p' src/scripts/partition_model/features.py

# Divergência do bloco D (achado 2)
sed -n '323,335p' docs/PLANO_H9_contribuicao_tese.md
sed -n '252,261p' src/scripts/partition_model/features.py
```
