# ConvNeXt com alvo de *regret* — hipótese REFUTADA, e a família de pixels fechada

**Data:** 2026-07-26
**Pergunta.** A tese usa o ConvNeXt substituto como **teto do domínio de pixels**, e dele
extrai a afirmação de que *"os pixels saturam na variância"*. Esse modelo foi treinado com
**entropia cruzada sobre rótulos duros** — isto é, otimizando acurácia por nó — quando a
própria tese demonstrou (Approach B, `RESULTADOS_approachB.md §6`) que **acurácia por-nó é
mau proxy do BD×tempo real de um podador**. O alvo correto, o *regret* do crivo A5, existe
(`train_regret.py`) mas **nunca havia sido aplicado a pixels**. Este experimento aplica.

**Hipótese:** treinar o ConvNeXt contra o *regret* levanta o teto do domínio de pixels.

**Scripts:** `train_surrogate_regret.py` (retreino) · `oracle_regret.py::score_surrogate`
(nova perna do crivo)
**Artefatos:** `results/models/surrogate_regret/`, `results/models/oracle_regret_convnext/`
(o A5 original em `oracle_regret/` foi preservado, não sobrescrito)

---

## 1. Método

**O que muda: só o objetivo.** Perda = CE ponderada por nó com peso `1 + α·regret_rel`,
`α=3`, com `regret_rel` vindo de `regret.node_regrets` (a mesma fonte do crivo A5),
normalizado por nível pelo percentil 95 para que α signifique o mesmo em 64/32/16px. Seleção
pelo **mesmo** critério na validação, com parada antecipada.

**O que NÃO muda, deliberadamente:** arquitetura (`PartitionSurrogate`), `fusion_dim=128`
(lido do checkpoint do modelo original), formato de saída, e todo o caminho a jusante. A
comparação precisa isolar o objetivo.

**Avaliação.** Crivo A5 sobre a vara held-out completa (6 sequências, val+test), com as duas
pernas novas: `convnext_ce` (o modelo original) e `convnext_regret`. Sem o primeiro na mesma
vara, um ganho do segundo não seria atribuível ao objetivo.

**Sem re-extração.** O `.pkl` guarda a luma sem perdas (`round(pkl·255)` = quadro-fonte,
`maxdiff=0`); foi trabalho de GPU, não de re-codificação da UVG.

### 1.1 Dois confundidores removidos antes de medir

- **Capacidade.** O primeiro retreino usou `fusion_dim=256` (default do script novo) contra
  os 128 do original — um ganho seria confundido com capacidade dobrada. Refeito com 128.
  Subproduto: os dois chegaram a `val_wloss` praticamente igual (0,9235 com 256 contra
  0,9250 com 128, **0,16% de diferença**) → **o modelo não está limitado por capacidade**.
- **Seleção.** Uma versão anterior desta análise alegava que o checkpoint original fora
  colhido em sobreajuste. **É falso** — ver §4.

## 2. Resultado — o retreino PIOROU o modelo, em todos os pontos

`reg_frac` (% de sobrecarga RD) em `cost_red` casado; **menor é melhor**:

| cost_red | variance | **convnext_ce** | **convnext_regret** | pixels24 | H9a |
|---|--:|--:|--:|--:|--:|
| 5% | — | 0,0009 | 0,0033 | 0,0001 | 0,0000 |
| 10% | 0,0125 | 0,0026 | 0,0054 | 0,0009 | 0,0002 |
| 15% | 0,0275 | 0,0051 | 0,0080 | 0,0023 | 0,0005 |
| 20% | 0,0424 | 0,0105 | 0,0140 | 0,0057 | 0,0013 |
| 25% | 0,0573 | 0,0207 | 0,0219 | 0,0121 | 0,0036 |

| cost_red | 5% | 10% | 15% | 20% | 25% |
|---|--:|--:|--:|--:|--:|
| `convnext_regret` / `convnext_ce` | **3,80×** | 2,07× | 1,58× | 1,33× | 1,06× |

### 2.1 Por que a refutação é forte

O modelo foi **treinado para minimizar erro ponderado por regret** e **avaliado em regret**.
Objetivo e métrica alinhados — o caso mais favorável possível para a hipótese. Ainda assim
perdeu para a CE simples em **toda** a faixa, com a piora **maior na região conservadora**
(3,80× em 5% de redução de custo), que é exatamente onde um podador implantável opera.

Explicação plausível, registrada como **hipótese e não como causa provada**: ponderar por
*regret* concentra capacidade nos nós raros e caros, degradando a **calibração** no grosso da
distribuição — e a política de NONE-commit depende de `P(NONE)` bem calibrado em todos os
nós, não de ordenar bem os caros.

### 2.2 O achado colateral, mais relevante que o principal

A hierarquia medida, em custo casado:

> **H9a < pixels24 < convnext_ce < convnext_regret < variance**

O ConvNeXt de **28,1 M de parâmetros sobre pixels crus perde para o `pixels24`** — um MLP
pequeno sobre 24 atributos manuais do mesmo domínio (0,0121 contra 0,0207 em 25% de redução
de custo, ~1,7×). Isso **abala o enquadramento de "teto do domínio de pixels"**: o modelo que
a tese usa como teto não é sequer o melhor modelo de pixels disponível nesta métrica.

## 3. Ressalvas

- **A curva da variância é grosseiramente amostrada nesta faixa.** A grade de τ salta de
  `cost_red` 6,14 (τ=0,99) para 39,46 (τ=0,97): **todos** os valores entre 6% e 39% são
  interpolados através de um único vão. A leitura "os modelos de pixels batem a variância por
  2–5×" repousa nessa interpolação e deve ser lida com reserva.
- **O crivo não adjudica.** Ele próprio documenta divergir do encoder no único par com chão
  limpo (GNN vs H9a). Serve para eliminar candidatos, não para coroar vencedores.
- **α=3 é um ponto único.** Outros pesos não foram varridos. Dado que a refutação é
  consistente em toda a faixa e maior onde mais importa, varrer α seria perseguir um
  resultado já claro — mas registre-se que não está excluído.
- **O crivo mede contra o ótimo RDO cpu-used=0**, não o custo cpu≥1 da implantação.

## 4. Correção de registro

Uma versão anterior desta análise (e de `ANDAMENTO_tese.md §0.4`, commit `0122b53`) alegava
um **segundo defeito** no ConvNeXt original: checkpoint selecionado por macro-F1 na época 27,
com a perda de validação 15% acima do mínimo. **É falso**, e vinha de leitura parcial do CSV.
Nas 30 épocas completas de `results/models/surrogate_real/metrics.csv`:

| | valor | época |
|---|--:|--:|
| mínimo de `val_loss` | 1,7999 | **13** |
| máximo de macro-F1 | 0,2034 | **13** |
| checkpoint salvo | — | **13** |

**Os dois critérios concordam e a seleção estava correta.** O modelo original é bem
selecionado, apenas **fraco em absoluto** (macro-F1 0,203) e treinado contra o objetivo
errado. O experimento se justificava por esse motivo único — e o motivo caiu.

## 5. Veredito — a família de pixels está fechada

O portão pré-registrado era "superar a variância no crivo A5". O `convnext_regret` supera —
**mas o `convnext_ce` também, e por mais**. Passar o portão sem superar a própria linha de
base torna o resultado inútil para prosseguir: gastar ~4h de replay H8 no encoder para medir
um modelo já pior offline que seu antecessor não se justifica.

**O alvo de *regret* não levanta o teto de pixels.** Somado ao que já estava medido, a
família baseada em pixels encerra assim:

| via | estado |
|---|---|
| ConvNeXt substituto (CE) | medido; teto declarado, mas **inferior ao `pixels24`** |
| ConvNeXt com alvo de *regret* | **refutado** — pior que a CE em toda a faixa |
| `pixels24` (estudante de 24 atributos) | melhor modelo de pixels medido; ainda **muito atrás do H9a** |
| Approach B / GNN (estrutural) | negativo no encoder (~2× pior), apesar de vencer offline |
| Bloco D — SATD do bloco-**fonte** | nulo no Gate 2; e **não era o atributo especificado** (ver auditoria de 26/07) |
| Bloco D' — predizibilidade intra a partir dos **vizinhos** | **portão não passa** (26/07): nulo em 16 px, com 15 855 nós |

Nenhuma via de pixels **aprendida de ponta a ponta** foi implantada.

> **Correção (2026-07-26).** Uma versão anterior deste parágrafo afirmava que o podador
> implantado "usa contexto de taxa-distorção, **não pixels**". **É falso.** O H9a é
> `A+B+C`, e o bloco `A` (índices 0–23) é rotulado no próprio código como
> `A 0..23 pixels` (`features.py:197`): variância global e por quadrante, gradientes
> h/v, orientação, densidade de bordas, contraste com pai e irmãos. **Vinte e quatro
> dos 36 atributos do H9a são descritores de luma**, e nenhum é uma grandeza de custo
> RD — as de custo RD são o bloco `E`, que define o H9c, rejeitado. Ver
> `RESULTADOS_auditoria_dominio_pixels.md`.

A leitura correta da hierarquia não é "contexto RD vence pixels", e sim: **descritores
manuais compactos de luma vencem uma rede convolucional profunda sobre pixels crus**, e
o que separa o campeão do resto é **contexto causal de vizinhança** (12 atributos de
custo ~zero) somado a esses descritores. O `pixels24` é literalmente o H9a menos os
blocos B e C. Retornos marginais medidos no mesmo crivo, em `cost_red` 25%:

| passo | `reg_frac` | ganho marginal |
|---|--:|--:|
| `variance` (1 descritor de luma) | 0,0573 | — |
| `pixels24` (+23 descritores de luma) | 0,0121 | **4,7×** |
| `convnext_ce` (+28,1 M parâmetros sobre pixels crus) | 0,0207 | **0,6× (pior)** |
| `H9a` (`pixels24` + 12 de vizinhança/quant/posição) | 0,0036 | **3,4×** |

O H9a domina todas as vias aprendidas no crivo por 3 a 6× — **sendo ele próprio,
majoritariamente, um modelo de pixels**.

### 5.1 O que isto NÃO resolve

A afirmação *"os pixels saturam na variância"* **continua sem base firme**, agora por um
motivo mais bem caracterizado: a ablação que a sustenta é de **2 quadros numa sequência** e
o crivo A5 a **contradiz** (os modelos de pixels batem a variância por 2–5×, ainda que sob
interpolação grosseira). Este experimento **não** resolve a contradição — ele apenas remove
a hipótese de que o objetivo de treino a explicaria.

Resolver exigiria uma ablação no encoder com ≥10 quadros e ≥2 sequências. O **E5** rodou
em 2026-07-28 (`RESULTADOS_E5_ablacao_validacao.md`) e **não resolve esta contradição
específica**: seu braço `ml` é o estudante **H9a** de 36 atributos, não o `pixels24`, de
modo que ele arbitra *H9a × variância* (o H9a vence a tempo casado, por 4,6× e 1,85×) e
não *pixels24 × variância*, que é o par em disputa aqui.

A recomendação permanece, portanto, a mesma: a tese deve **reportar a hierarquia medida**
(H9a < pixels24 < ConvNeXt < variância no crivo; variância ≥ pixels na ablação de 2
quadros, em contradição declarada) em vez de afirmar a saturação como conclusão. O que
mudou é que o topo dessa hierarquia — o H9a — passou a ter confirmação **no codificador**,
e não só no crivo offline.

## 6. Reprodução
```bash
/workspace/build/venv-ml/bin/python src/scripts/partition_model/train_surrogate_regret.py
/workspace/build/venv-ml/bin/python src/scripts/partition_model/oracle_regret.py \
    --out-dir /workspace/results/models/oracle_regret_convnext
```
