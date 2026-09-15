# ICASSP — avaliação restrita às três sequências de teste (separação limpa treino/validação/teste)

**Data:** 2026-09-12
**Origem:** colocação do orientador Daniel Palomino, anterior ao controle de permutação
([[RESULTADOS_ICASSP_controle_permutacao_NC]]).
**Pedido:** recalcular Tabela I, Figura 2 e todos os resultados usando apenas as três sequências
reservadas para teste, já que as três de validação servem para escolher o checkpoint do ConvNeXt
e não deveriam compor o conjunto de avaliação final.
**Scripts:** `src/scripts/partition_model/oracle_regret.py` ·
`src/scripts/benchmark/plot_loss_frontier_icassp_fig2.py`
**Artefatos:** `results/models/oracle_regret_rpp_test3/` (fonte única do artigo a partir daqui)

---

## 1. Validação da colocação

Verificada no código, premissa por premissa.

- **O checkpoint do ConvNeXt é escolhido pela validação — confirmado.**
  `train_surrogate_regret.py:91` define `VAL_SEQS = ["HoneyBee", "FlowerPan", "Lips"]`, e as
  linhas 335–349 gravam `surrogate_regret_best.pt` na época de menor perda de validação.
  `oracle_regret.py` carrega exatamente esses `*_best.pt` para os braços ConvNeXt.
- **Essas sequências estavam na avaliação final — confirmado.** O `--seqs` padrão eram as seis.
- **As cinco rungs tabulares NÃO estão contaminadas.** `train_student` (`distill.py:98–119`)
  treina 30 épocas fixas, sem validação, sem parada antecipada e sem seleção de checkpoint. A
  afirmação central do artigo (BC contra BC+NC) já era limpa antes desta correção.
- **A premissa de que a inclusão favorecia o ConvNeXt está empiricamente invertida.** Ver §4.

O pedido procede pelo quarto argumento dele, que independe da direção: não é separação limpa e
confunde revisor.

## 2. Reprodução

```bash
build/venv-ml/bin/python src/scripts/partition_model/oracle_regret.py \
    --seqs Jockey RaceNight RiverBank \
    --out-dir /workspace/results/models/oracle_regret_rpp_test3
```

Conjunto de avaliação: **111.426 superblocos, 1.816.393 nós de decisão** (antes 226.447 e
3.808.703 sobre as seis). Treino segue nas mesmas dez sequências.

## 3. Tabela I nova

Penalidade de custo RD normalizada, 10^-4 %, média de 3 sementes. Entre parênteses, o valor
antigo sobre as seis sequências.

| Predictor | 10% | 15% | 20% | 25% | 30% |
|---|--:|--:|--:|--:|--:|
| Random control | 1610 (1963) | 2465 (2989) | 3331 (4066) | 4191 (5166) | 5122 (6342) |
| Variance | 13 (43) | 49 (127) | 96 (250) | 176 (374) | 298 (555) |
| ConvNeXt, plain CE | 35 (65) | 61 (105) | 88 (164) | 141 (272) | 221 (442) |
| ConvNeXt, width 256 | 49 (75) | 71 (122) | 113 (194) | 190 (294) | 285 (444) |
| BC (24) | 6 (5) | 11 (13) | 23 (28) | 50 (53) | 84 (91) |
| BC+shuffled NC (32) | 6 (5) | 11 (13) | 24 (27) | 46 (48) | 76 (86) |
| **BC+NC (32)** | **3** (2) | **5** (5) | **11** (15) | **27** (33) | **60** (69) |
| BC+CSP (28) | 6 (5) | 11 (12) | 28 (27) | 78 (64) | 152 (122) |
| BC+NC+CSP (36) | 2 (2) | 5 (6) | 14 (18) | 34 (40) | 68 (78) |

A linha `ConvNeXt, cost-sensitive` foi **removida do artigo** — ver §5.

## 4. O que mudou nas afirmações

**Reforçadas:**

- BC+NC sobre BC em 25%: **37,7% → 46,0%**.
- BC+NC sobre o controle de permutação em 25%: **30,4% → 41,3%**; por ponto, 57,4% / 54,8% /
  51,8% / 40,1% / 21,3%.
- A disjunção entre as faixas do controle e de BC+NC se mantém em 10, 15, 20 e 25% (sobrepõe só
  em 30%, como antes), e a sobreposição controle~BC vale nos cinco pontos.
- BC+CSP piora bem mais (78 e 152 em 25 e 30%, com faixa de 40,1 a 134,6), o que fortalece o
  argumento de que somar colunas não basta.

**Quebradas:**

- **O treino sensível a custo deixa de vencer.** Nas três de teste o CE puro vence nos cinco
  pontos (141 contra 185 em 25%). A afirmação anterior — *"cost-sensitive training consistently
  improves the ConvNeXt"* — inverte por completo.
- **A vantagem sobre a variância cai pela metade:** 7,0× → **3,5×** (176 contra 50 em 25%).
- **BC+NC deixa de ser o menor em 10%:** BC+NC+CSP fica em 2,16 contra 2,55.

**Premissa invertida.** Razão ConvNeXt/BC em 25%, com BC limpo nas duas rodadas:

| | 6 seqs | só teste |
|---|--:|--:|
| plain CE | 5,08× | 2,80× |
| width 256 | 5,51× | 3,77× |
| cost-sensitive | 4,09× | 3,68× |

Removendo a validação, o ConvNeXt fica **relativamente melhor**, não pior — o mesmo valendo para
a variância. Ou seja, a avaliação anterior **não** era conservadora como se supunha: a correção
custa margem sobre as baselines e paga com um resultado principal maior.

**Ressalva de medição:** duas coisas mudaram juntas (saiu o vazamento, mudou o corpus), então a
inversão não está isolada. A leitura mais provável é que HoneyBee/FlowerPan/Lips são conteúdo
onde o ConvNeXt vai mal por razão de textura, e esse efeito superou a vantagem da seleção de
checkpoint. Isolar exigiria uma rodada só nas três de validação, comparando a razão ConvNeXt/BC
nos dois corpora — não executada.

## 5. Decisão sobre o braço custo-sensível — revertida

**Primeira decisão (descartada):** remover o braço da Tabela I e da Figura 2, sob o argumento de
que a regra "reportar a baseline de pixel mais forte" continuava sendo aplicada — antes apontava
para o custo-sensível, agora para o CE puro.

**O orientador apontou o defeito, e ele procede:** essa regra é decidida *olhando o conjunto de
avaliação*. Nos dois momentos, o que se reporta foi escolhido pelo resultado — antes porque o
custo-sensível vencia nas seis sequências, agora porque o CE puro vence nas três de teste. É
exatamente a prática que o artigo critica na literatura.

**Decisão final:** a linha volta à Tabela I (43, 65, 95, 185, 275) e a curva volta à Figura 2. O
§IV-B volta a declarar os dois objetivos em pé de igualdade, e o §V reporta o resultado negativo
com explicação técnica: o objetivo alinha a perda de treino à métrica de avaliação, e a sua falha
em transferir é informativa — a ponderação concentra a perda nos poucos nós de maior penalidade
RD, o que pode ser atribuído a estrutura específica do conteúdo que não generaliza para sequências
não vistas. A explicação leva *hedge* por ser hipótese, não medida.

Custo de página: a linha e o parágrafo empurraram o Agradecimento para a página 5, que deve conter
apenas referências. Recuperado **sem perder conteúdo** — enxugando redundância de redação e
comprimindo a folga vertical da Figura 1, de 3,07 para 2,64 polegadas.

## 6. Limitações

- **Conjunto de avaliação pela metade:** 3 sequências em vez de 6. Na prática as faixas entre
  sementes ficaram semelhantes às anteriores, com uma exceção — BC+CSP explodiu para 40,1–134,6
  em 25%.
- **A inversão da premissa não foi isolada** (ver ressalva na §4).
- **Continua replay offline**, sem propagação de erro de reconstrução e sem BD-rate.

## 7. Efeito nos artefatos

- `plot_loss_frontier_icassp_fig2.py` passa a ler `oracle_regret_rpp_test3/frontier.csv`, perde a
  curva custo-sensível e confere 45 células contra a Tabela I nova.
- `results/models/oracle_regret_rpp/frontier.csv` (6 sequências) permanece no repositório como
  registro da rodada anterior, mas **não é mais a fonte do artigo**.
- Os números da §4 de [[RESULTADOS_ICASSP_controle_permutacao_NC]] referem-se à rodada de 6
  sequências e estão superados por esta; o método e as conclusões daquele documento seguem
  válidos.
