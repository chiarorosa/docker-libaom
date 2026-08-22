# RPP — escada informacional sob receita fixa, e a remedição da hierarquia do crivo

**Data:** 2026-08-20
**Estado:** contribuição metodológica + correção de três números da tese. Nenhum encode novo.
**Vara:** validação + teste held-out (HoneyBee, FlowerPan, Lips, Jockey, RaceNight,
RiverBank), **226.447 superblocos e 3.808.703 nós de decisão**, custo RD total 1,28×10¹⁴.
Modelos treinados nas dez sequências restantes.
**Artefatos:** `results/models/oracle_regret_rpp/{frontier,ranking}.csv` e `report.md`;
`results/models/rpp_ladder/` (12 bundles + `training.csv`);
`results/models/surrogate_ce_h9/` e `results/models/surrogate_ce_h9_f256/`.
**Scripts:** `src/scripts/partition_model/rpp_ladder.py` (novo),
`oracle_regret.py` (grade densa da variância + pernas RPP e ConvNeXt corrigidas),
`train_surrogate_regret.py --alpha 0`.

---

## 1. A pergunta

A tese sustenta uma escada de informação para a decisão de particionamento intraquadro:
variância isolada → vinte e quatro colunas compactas → mais doze colunas de contexto causal,
com uma rede convolucional profunda sobre pixels crus medida ao lado. Os ganhos marginais
registrados são de 4,7× e 3,4×, com a rede piorando por 0,6×.

Três defeitos de desenho tornam essa escada mais fraca do que aparenta, e este documento
os remove.

**D1 — a linha de base era interpolada.** A grade de τ compartilhada salta, para a
variância, de 6,14% para 39,46% de redução de custo. Todo valor entre eles, inclusive os
25% em que a escada é lida, vinha de interpolação através de um único vão.

**D2 — o procedimento de treino era um confundidor.** O braço de vinte e quatro colunas
(`pixels24`) é o `student_real`, **destilado** do ConvNeXt (`distill.py`, combinação convexa
de entropia cruzada e divergência contra o mestre). O braço de trinta e seis
(`student_h9a`) é entropia cruzada direta, sem mestre. A comparação entre eles mede
conjunto de atributos **e** objetivo de treino, simultaneamente.

**D3 — os doze atributos causais nunca foram separados.** Blocos B (vizinhança de
particionamento) e C (quantização efetiva, posição no quadro, profundidade) sempre foram
medidos juntos, de modo que "o contexto causal ajuda" não dizia *qual* contexto.

A auditoria do corpus do braço profundo, um quarto defeito, está em
[`RESULTADOS_auditoria_convnext_corpus.md`](RESULTADOS_auditoria_convnext_corpus.md) e é
pressuposta aqui.

## 2. O desenho

**Grade densa da variância (D1).** `oracle_regret.py` passa a aplicar uma grade própria de
26 valores de τ ao braço de variância, com quatorze pontos adicionais em (0,96; 0,999). O
ponto τ = 0,980 entrega **25,20% de redução de custo medida**, de modo que o valor lido a
25% deixa de depender da interpolação. Nenhum outro braço precisa disso: todos já possuíam
pontos densos na faixa.

**Escada sob receita única (D2, D3).** `rpp_ladder.py` lê os superblocos de treino **uma
única vez** e treina os quatro degraus com hiperparâmetros idênticos — topologia 64-32,
trinta épocas, taxa 1×10⁻³, 3.000 superblocos por `.pkl`, sem ponderação de classe —
variando **apenas o subconjunto de colunas** do vetor de trinta e seis:

| degrau | colunas | conteúdo |
|---|--:|---|
| A | 0–23 | 21 descritores de luminância + índice de quantização + posição no superbloco |
| A+B | 0–31 | + vizinhança causal de particionamento (acima e à esquerda) |
| A+C | 0–23, 32–35 | + passo de dequantização efetivo, posição no quadro, profundidade |
| A+B+C | 0–35 | o vetor livre completo |

Cada degrau é treinado com **três sementes**, semeadas por (degrau, nível) para que a
inicialização de um degrau não dependa de quantos rodaram antes dele. Nada no caminho de
treino era semeado antes deste trabalho.

**Nota de composição, que a leitura exige.** O degrau A **já contém** o índice de
quantização (`q_norm`, índice 17) e a posição no superbloco (índices 22 e 23). Logo, o que
B e C acrescentam **não pode ser atribuído ao modelo passar a dispor da quantização**: ela
estava presente dos dois lados. Pela mesma razão a comparação com o ConvNeXt é casada nesse
eixo, uma vez que a entrada da rede inclui um plano constante de quantização.

**Reprodução (contêiner):**

```bash
build/venv-ml/bin/python src/scripts/partition_model/rpp_ladder.py \
  --out-dir results/models/rpp_ladder --seeds 0 1 2

build/venv-ml/bin/python src/scripts/partition_model/train_surrogate_regret.py \
  --alpha 0 --fusion-dim 128 --out-dir results/models/surrogate_ce_h9
build/venv-ml/bin/python src/scripts/partition_model/train_surrogate_regret.py \
  --alpha 0 --fusion-dim 256 --out-dir results/models/surrogate_ce_h9_f256

build/venv-ml/bin/python src/scripts/partition_model/oracle_regret.py \
  --out-dir results/models/oracle_regret_rpp
```

## 3. A fronteira medida

`reg_frac`, em **por cento do custo RD total** do conjunto de nós de decisão; menor é
melhor. Degraus RPP são média de três sementes.

| braço | 5% | 10% | 15% | 20% | 25% | 30% |
|---|--:|--:|--:|--:|--:|--:|
| aleatório | 0,0980 | 0,1963 | 0,2989 | 0,4066 | 0,5166 | 0,6342 |
| variância isolada (grade densa) | 0,0007 | 0,0043 | 0,0127 | 0,0250 | **0,0374** | 0,0555 |
| ConvNeXt-CE, 28,1 M, α = 0 | 0,0029 | 0,0065 | 0,0105 | 0,0164 | 0,0272 | 0,0442 |
| ConvNeXt-CE, fusão 256 | 0,0036 | 0,0075 | 0,0122 | 0,0194 | 0,0294 | 0,0444 |
| ConvNeXt, α = 3 | 0,0033 | 0,0054 | 0,0080 | 0,0140 | 0,0219 | 0,0379 |
| **A** (24 col) | 0,0002 | 0,0005 | 0,0013 | 0,0028 | **0,0053** | 0,0091 |
| **A+B** (32 col) | 0,0001 | 0,0002 | 0,0005 | 0,0015 | **0,0033** | 0,0069 |
| **A+C** (28 col) | 0,0002 | 0,0005 | 0,0012 | 0,0027 | 0,0064 | 0,0122 |
| **A+B+C** (36 col) | 0,0000 | 0,0002 | 0,0006 | 0,0018 | 0,0040 | 0,0078 |
| *[legado] `pixels24` destilado* | 0,0001 | 0,0009 | 0,0023 | 0,0057 | 0,0121 | 0,0215 |
| *[legado] `convnext_ce` contaminado* | 0,0009 | 0,0026 | 0,0051 | 0,0105 | 0,0207 | — |
| *[legado] `H9a` implantado* | 0,0000 | 0,0002 | 0,0005 | 0,0013 | 0,0036 | 0,0073 |

Valores por semente a 25%, que são o que sustenta as separações da §4:

| degrau | s0 | s1 | s2 | média |
|---|--:|--:|--:|--:|
| A | 0,0057 | 0,0049 | 0,0054 | 0,0053 |
| A+B | 0,0034 | 0,0030 | 0,0036 | 0,0033 |
| A+C | 0,0090 | 0,0056 | 0,0045 | 0,0064 |
| A+B+C | 0,0040 | 0,0040 | 0,0040 | 0,0040 |

## 4. O que fica estabelecido

**4.1 A grade grosseira inflava a variância em 1,53×.** Medida, a variância entrega 0,0374
a 25% de redução de custo casada, contra os 0,0573 interpolados. O ponto τ = 0,980, com
25,20% de redução, é medido.

**4.2 O procedimento de treino valia de 2,0× a 2,4×.** Com as **mesmas vinte e quatro
colunas**, o braço destilado marca 0,0121 e o braço de receita fixa marca 0,0053 a 25% —
uma razão de 2,25×, que a hierarquia anterior atribuía aos atributos. O crivo é, portanto,
sensível ao procedimento de treino, e uma escada de informação que não o fixa não mede
informação.

**4.3 O bloco B carrega o sinal, e a separação é robusta.** De A para A+B o ganho é de
**1,60×** a 25% (1,88× a 20%; 1,31× a 30%), e **as três sementes de A+B superam as três
sementes de A** — pior A+B em 0,0036 contra melhor A em 0,0049. Não há sobreposição.

**4.4 O bloco C não acrescenta, e desestabiliza.** A+C marca 0,0064 de média contra 0,0053
de A, mas com dispersão entre sementes de 0,0045 a 0,0090, o dobro da de A. As distribuições
se sobrepõem: o enunciado defensável é que **C não acrescenta informação decisória**, e não
que ele prejudique quando isolado.

**4.5 C prejudica quando somado a B, e aí sim de forma consistente.** A+B marca 0,0033 e
A+B+C marca 0,0040, com **as três sementes de A+B superando as três de A+B+C** e A+B+C
exibindo dispersão nula. O subconjunto de trinta e duas colunas domina o de trinta e seis
**de 14,2% de redução de custo em diante**, e abaixo desse ponto a ordem se inverte, com as
duas a menos de 0,00005 ponto percentual uma da outra. O vetor implantado não é, offline, o
melhor dos quatro no regime lido.

> **Correção de registro (2026-08-22).** Esta subseção afirmava dominância em **todos** os
> pontos da fronteira. A afirmação vinha da tabela arredondada, em que A+B e A+B+C leem
> ambas 0,0002 a 10% de redução de custo, quando são 0,000235 e 0,000187. O cruzamento foi
> localizado ao gerar a Figura 2 do artigo ICASSP, que plota a fronteira contínua.

**4.6 A escada corrigida.** Sob receita fixa e grade densa, a 25%: variância 0,0374 →
A 0,0053 (**7,0×**) → A+B 0,0033 (**1,60×**). Os números anteriores, 4,7× e 3,4×, não se
sustentam: o primeiro repousava em interpolação e o segundo em confundimento de treino.

**4.7 O braço profundo perde por cerca de 4× a 5×, sob tratamento justo.** Com corpus
correto, sem vazamento, capacidade dobrada e o melhor dos dois objetivos disponíveis, o
ConvNeXt de 28,1 milhões de parâmetros marca 0,0219 contra 0,0053 do degrau A — **4,1× pior
com 2.481× mais parâmetros**. Dobrar a largura de fusão piora (0,0294 contra 0,0272), o que
confirma que a capacidade não é a restrição.

**4.8 A conclusão da tese sobre o alvo de perda de otimalidade se inverte.** Com corpus,
escalonamento e critério de seleção idênticos, α = 3 é **melhor** que α = 0 em todos os
pontos de 10% para cima (0,0219 contra 0,0272 a 25%). A tese registra o oposto — piora de
1,06× a 3,80× —, e aquele registro é artefato do confundimento de corpus documentado na
auditoria. A hipótese "o alvo de perda de otimalidade levanta o teto de pixels" foi
**confirmada fracamente**, e não refutada: ela levanta o teto, mas não o suficiente para
que a via de pixels compita.

**4.9 O veredito sobre a família de pixels sobrevive.** Nenhuma via de pixels aprendida de
ponta a ponta compete com a representação compacta: mesmo o melhor braço profundo perde por
4,1×, com três ordens de grandeza a mais de parâmetros. O que muda é uma das razões
registradas para o fechamento, não o fechamento.

## 5. Limitações

- **O crivo não adjudica.** No único par com chão de codificador limpo (GNN contra H9a) ele
  diverge do encoder. Ele elimina candidatos inferiores; não coroa vencedores. Nenhuma das
  razões desta análise é previsão de taxa BD ou de tempo de parede.
- **O denominador de `reg_frac` soma custos mutuamente excludentes** — os três níveis
  recobrem a mesma área da imagem, e um superbloco contribui com vinte e um nós dos quais só
  um subconjunto disjunto sobrevive. Isso deprime a razão para um mesmo numerador, e só
  isso: não autoriza compará-la em magnitude com perda medida por recodificação. A grandeza
  é legítima apenas em regime ordinal.
- **A superioridade de A+B sobre A+B+C não foi verificada no codificador.** Fechá-la
  exigiria uma campanha análoga à do E5, com a ordem de 144 codificações.
- **Três sementes** dão amplitude, não intervalo de confiança. As separações reportadas como
  robustas na §4 são separações de suporte (nenhuma sobreposição entre os três valores de
  cada braço), e não testes de hipótese.
- **α = 0 e α = 3 são dois pontos**; a curva em α não foi varrida.
- **Os rótulos são o ótimo RDO a `cpu-used=0`**, não o custo de um regime de implantação
  mais rápido.
- **`--alpha 0` é entropia cruzada simples**, ao passo que o objetivo do `surrogate_real`
  original incluía rótulos suaves sensíveis a custo, ponderação por frequência inversa e
  suavização. O braço novo não é o original corrigido; é um braço de entropia cruzada limpo.

## 6. Documentos da tese a corrigir

| documento | o que muda |
|---|---|
| `R1_dominio_pixels.md` §1.4 | hierarquia a 25%, contagem de nós, e a mistura de duas execuções do crivo no mesmo parágrafo |
| `R1_dominio_pixels.md` §1.5 | a piora de 1,06×–3,80× atribuída ao objetivo |
| `R1_dominio_pixels.md` §1.6 | os ganhos marginais de 4,7× e 3,4× |
| `R1_dominio_pixels.md` §1.7, §1.8 | "cinco tentativas independentes negativas" |
| `M4_atributos_e_politica.md` §4.3 | a releitura da hierarquia |
| `M6_modelos_e_atribuicao.md` §6.1 | a ablação de objetivo e o controle de capacidade |
| `T_RESULTADOS_tabelas.md` | a linha da hierarquia do crivo |
| `A1_INDICE_evidencias.md` | as duas linhas do ConvNeXt e a do fechamento da família |
| `R5`, `R6`, `INVENTARIO_solucoes.md`, `SINTESE_resultados_metodologia.md`, `ANDAMENTO_tese.md` | a contagem de tentativas e a razão do fechamento |

> **Procedência.** Fronteira: `results/models/oracle_regret_rpp/frontier.csv` (26 braços;
> `variance` com 26 valores de τ). Bundles: `results/models/rpp_ladder/*/students.pt` e
> `training.csv`. Braços profundos: `results/models/surrogate_ce_h9/` e
> `surrogate_ce_h9_f256/`, com registros em `results/models/surrogate_ce_h9_train.log` e
> `surrogate_ce_h9_f256_train.log`. Subconjuntos de colunas:
> `src/scripts/partition_model/features.py`, `RPP_SUBSETS`. Grade densa:
> `oracle_regret.py`, `VARIANCE_TAUS`. Definição de `reg_frac` e do modelo de custo:
> `regret.py` e `simulate_pruning.py` (`CANDS`, `node_cost`). Auditoria do corpus do braço
> profundo: `RESULTADOS_auditoria_convnext_corpus.md`.
