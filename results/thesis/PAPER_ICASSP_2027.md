# Attribution Before Acceleration: A Regret-Weighted Oracle Sieve for Learned AV1 Intra-Partition Pruning

**[completar: nomes dos autores, afiliação e financiamento, conforme exigido pelo CFP]**

## Abstract

Learned pruning of the recursive intra-frame partition search in AV1 is
commonly evaluated by a single number: the speedup obtained once the pruner
is deployed under an aggressive policy. This number is not evidence of
learning, because a pruner is inseparably a scoring source plus a policy, and
any score — including an uninformative one — accelerates the encoder once the
policy is aggressive enough. This paper isolates the scoring source from the
policy on a machine-learning-native testbed of sixteen 4K UVG sequences,
split by sequence into ten training, three validation, and three held-out
test sequences, with 26.98 million logged decision-tree samples. A
rate-distortion-regret-weighted oracle sieve shows that a 28.1-million
parameter convolutional network over raw luma loses to a 24-descriptor
perceptron built on the same luma, and that twelve near-zero-cost causal
attributes of partition neighborhood, quantization, and position buy a
further 3.4-fold reduction in regret over that perceptron. A controlled
ablation over message-passing depth shows a structured graph model beating
the deployed perceptron by 28 percentage points in the oracle sieve, yet
losing to it by a factor of two in bit-rate under matched policy inside the
real encoder — an inversion that reduces the oracle sieve to a filter,
never an arbiter, of competing models. A matched-time ablation inside the
encoder corroborates the attribution of the deployed model's gain on two
validation sequences, with the strict pre-registered criterion met on one of
two comparable sequences.

**Index Terms** — AV1, intra-frame coding, partition search pruning,
learned heuristics, attribution, surrogate metrics, oracle simulation,
rate-distortion optimization, graph neural networks.

---

## 1. Introdução

A decisão de particionamento de blocos do AV1 em predição intraquadro é
tomada por uma busca recursiva que avalia, em cada nó da árvore
quaternária, até dez formas de partição candidatas pelo seu custo de
taxa-distorção (do inglês *rate-distortion* – RD), mantendo a de menor
custo e descendo quando a divisão vence. Esta busca é exaustiva por
construção e responde por um dos maiores custos computacionais do
codificador de referência. A poda aprendida do espaço de busca — um modelo
de aprendizado de máquina consultado em cada nó, que elimina candidatos
improváveis antes de o codificador pagar o custo de avaliá-los — é a
heurística proposta pela literatura recente para reduzir esse custo com o
menor impacto possível na eficiência de compressão.

A avaliação usual desta heurística repousa sobre um único número: a
aceleração medida quando o podador é embarcado sob uma política agressiva.
Este número, isoladamente, não é evidência de que o modelo aprendeu algo.
Um podador aprendido é composto por duas peças separáveis — uma **fonte de
pontuação**, que é o modelo, e uma **política**, que converte a pontuação
em ações de poda sobre o espaço de busca —, e qualquer pontuação, inclusive
uma pontuação sem informação alguma, produz redução de tempo quando
alimentada a uma política suficientemente agressiva, uma vez que é a
política, e não a pontuação, que remove candidatos da busca. A ausência
desta distinção é uma fragilidade recorrente na literatura de poda
aprendida de codificadores de vídeo, e é o problema central que este artigo
enfrenta por medição, e não por argumentação.

Este artigo apresenta um conjunto de desenhos experimentais que separa o
mérito do modelo do mérito do invólucro que o executa, aplicado sobre o
domínio de pixels e sobre o contexto de taxa-distorção barato da árvore de
particionamento do AV1. São apresentados, nesta ordem: o conjunto de dados
e a partição congelada por sequência que sustentam toda a medição; um crivo
ponderado por perda de otimalidade (do inglês *regret*), usado para
triar candidatos ao custo de processamento gráfico, em vez do custo de
codificação real; a hierarquia medida sobre este crivo, com a auditoria de
composição que corrige a leitura ingênua desta hierarquia; a validade do
próprio crivo como indicador substituto, estabelecida por uma ablação
controlada que expõe uma inversão de ordenação entre a simulação e o
codificador real; a reformulação do problema por regressão, contraposta à
classificação; e, por fim, a confirmação dentro do codificador, sob
política idêntica e a tempo casado.

O cenário experimental deste artigo é o **universo do próprio aprendizado
de máquina**: dezesseis sequências UVG em 4K, particionadas por sequência
sem vazamento, com simulação oráculo, crivo ponderado por perda de
otimalidade e confirmação no codificador sob política casada. Este cenário
não é a grade das condições comuns de teste (CTC) da indústria, e nenhum
resultado de implantação sob protocolo CTC, nem contribuição marginal de
podadores empilhados, é objeto deste artigo. O que este artigo estabelece é
a validade — e os limites — da atribuição de um ganho medido ao
aprendizado, e não à magnitude prática desse ganho contra o botão de
velocidade do codificador.

---

## 2. Conjunto de dados, instrumentação e partição congelada por sequência

O conjunto de dados canônico foi extraído do codificador de referência
libaom v3.10.0, instrumentado para registrar, em cada nó da busca de
particionamento intraquadro, os pixels de luminância do bloco, o rótulo de
referência da decisão de taxa-distorção completa, o contexto de
particionamento dos vizinhos causais acima e à esquerda, o passo de
dequantização e a posição do nó. A extração foi executada integralmente sob
`cpu-used=0`, o único regime em que todas as dez formas de partição são
exploradas por busca RD completa. A contaminação de rótulos por *presets*
mais rápidos foi verificada empiricamente: a `cpu-used=6`, as seis classes
de partição estendida (`HORZ_A`, `HORZ_B`, `VERT_A`, `VERT_B`, `HORZ_4` e
`VERT_4`) apresentam contagem zero; a `cpu-used=8`, o modo *All-Intra*
abandona o caminho RD, e nada é registrado. A densidade de amostragem
confirma o mesmo efeito: cerca de 125.775 amostras por quadro a
`cpu-used=0`, contra cerca de 34.593 a `cpu-used=3` — uma razão próxima de
3,6 vezes.

O corpus reúne **dezesseis sequências** do conjunto UVG em resolução 4K
(3840×2160, 8 bits, 4:2:0), codificadas em **quatro pontos de
quantização** (`cq-level` 20, 32, 43 e 55), sobre **cinco quadros**
amostrados de forma temporalmente uniforme ao longo de cada clipe. O
conjunto resultante contém **26,98 milhões de amostras**, das quais
**10,07 milhões são nós de decisão** (dimensão de bloco em {16, 32, 64}
amostras); os 16,91 milhões restantes são folhas terminais de 8×8 amostras,
de rótulo constante, e ficam fora da modelagem. O desbalanceamento de
classe é acentuado e depende do tamanho de bloco: nos nós de 64×64
amostras, `SPLIT` responde por 64,12% e `NONE` por 28,84%; nos de 32×32,
`NONE` sobe para 47,72% e `SPLIT` cai para 28,86%; nos de 16×16, `NONE`
domina com 73,46%, contra 4,40% de `SPLIT`.

A partição em treino, validação e teste reservado foi feita **por
sequência**, sem vazamento de conteúdo, e congelada antes de qualquer
medição, conforme resume a Tabela 1.

**Tabela 1** — Composição do conjunto de dados por partição, dezesseis
sequências UVG 4K, quatro pontos de quantização.

| partição | sequências | amostras (M) | nós de decisão (M) |
|---|---|---:|---:|
| treino | Beauty, Bosphorus, CityAlley, FlowerFocus, FlowerKids, ReadySetGo, ShakeNDry, SunBath, Twilight, YachtRide (10) | 15,76 | 6,09 |
| validação | HoneyBee, FlowerPan, Lips (3) | 5,89 | 2,07 |
| teste reservado | Jockey, RaceNight, RiverBank (3) | 5,32 | 1,91 |
| **total** | **16** | **26,97** | **10,07** |

O total da coluna de amostras é a soma das três parcelas arredondadas, e por isso registra 26,97 milhões, ao passo que a contagem exata do conjunto, citada no parágrafo anterior, é de 26,98 milhões. A diferença é de arredondamento, e não de contagem.

> **Nota de procedência.** Dado de origem: `results/dataset_h9/manifest.csv`
> (64 linhas). Script: `src/scripts/partition_dataset/rebuild_manifest_stats.py`.
> Fonte: `results/thesis/M2_instrumentacao_e_dataset.md` §2.4; partição
> congelada conforme `results/thesis/M3_protocolo_avaliacao.md` §3.2.

Um defeito de representação de dados atravessou a cadeia exploratória
inicial deste projeto e é registrado aqui como lição metodológica. A
luminância era gravada como `float32` em [0,1], ao passo que os
consumidores em Python assumiam `uint8` em [0,255], o que produzia
atributos truncados a zero e entradas de rede quase nulas — os modelos
daquela cadeia foram, por consequência, treinados sobre imagem em branco.
O dado bruto estava íntegro (`round(luma·255)` reproduz o quadro-fonte com
diferença máxima nula); apenas os consumidores estavam errados. A correção
institui uma função única de desnormalização e uma asserção de guarda que
interrompe a execução caso a luminância montada não apresente variância
real, guarda hoje ativa em todos os *scripts* consumidores do conjunto de
dados. Todos os números deste artigo são posteriores a esta correção.

Este cenário experimental — dezesseis sequências UVG 4K, cinco quadros por
sequência, partição 10/3/3 — é distinto do cenário das condições comuns de
teste da indústria (oito sequências da Classe A1, quinze quadros), e nenhum
resultado obtido sob este segundo protocolo é apresentado neste artigo.

> **Nota de procedência.** `results/thesis/M2_instrumentacao_e_dataset.md`
> §2.2, §2.3, §2.4 e §2.6; `results/thesis/R1_dominio_pixels.md` §1.2.

---

## 3. O crivo ponderado por perda de otimalidade

Cada campanha de codificação real custa horas de processamento, de modo que
toda comparação entre candidatos a podador anterior à integração em C é
resolvida por **simulação oráculo**: a política de poda é reproduzida
sobre o dado de referência do conjunto reservado, nó a nó, sem codificar um
único quadro. Duas grandezas complementares compõem o crivo empregado neste
artigo.

A **redução de custo de busca**, `cost_red`, é o eixo de operação comum
entre candidatos. O custo de um nó de dimensão *n* é o produto do número de
formas candidatas por *n*² — nove candidatos para blocos de 64, 32 e 16
amostras, e quatro para blocos de 8 amostras —, e `cost_red` é a diferença
percentual entre o custo somado sobre os nós que a política visita e o
custo da busca completa. Esta grandeza permite **casar** o ponto de
operação entre modelos concorrentes antes de comparar o dano que cada um
provoca.

A **fração de perda de otimalidade**, `reg_frac`, é a métrica de dano. No
numerador, a perda de otimalidade absoluta de um nó é o sobrecusto de
taxa-distorção pago por comprometê-lo com `PARTITION_NONE` em vez de tomar
a subárvore ótima — grandeza não negativa, nula sempre que a decisão
correta já era `PARTITION_NONE`. No denominador está o custo de
taxa-distorção total do conjunto de nós de decisão, **acumulado antes de
qualquer limiar** e mantido constante ao longo de toda a varredura, o que
torna a grandeza legível como fronteira: ela tende a zero quando a poda
tende a zero.

$$
\text{reg\_frac}(\tau) = \frac{\sum_{n \,\in\, \text{podados}(\tau)} \text{regret\_abs}(n)}{\sum_{n \,\in\, \text{nós de decisão}} \text{RDcost}(n)} \times 100\%
$$

O crivo foi aplicado a **792.840 nós de decisão** das seis sequências de
validação e de teste, sobre modelos treinados apenas nas dez sequências de
treino restantes, e a grandeza é lida a **redução de custo casada**, de
modo que todos os candidatos sejam comparados no mesmo ponto de operação.

Duas ressalvas de escopo acompanham o crivo em todo o restante deste
artigo. A primeira é de **magnitude**: a redução de custo simulada supera o
tempo de parede real por um fator historicamente próximo de **cinco vezes**
— 35% de redução de custo no oráculo corresponderam, em medição direta, a
cerca de 7% de tempo de parede —, de modo que nenhum valor de `reg_frac` ou
`cost_red` é apresentado como número absoluto de custo de codificação. A
segunda é de **amostragem**: a grade de limiares do braço de variância
salta de 6,14% para 39,46% de redução de custo, de modo que todos os
valores intermediários da curva da variância citados na Seção 4 são
interpolados através de um único vão. O crivo, por construção, **não
adjudica** — a Seção 5 mede diretamente essa limitação — e a sua função
declarada é eliminar candidatos inferiores a baixo custo, nunca coroar um
vencedor.

> **Nota de procedência.** `results/thesis/M6_modelos_e_atribuicao.md`
> §6.10; `results/thesis/M3_protocolo_avaliacao.md` §3.4.1;
> `results/thesis/R1_dominio_pixels.md` §1.4. Script:
> `src/scripts/partition_model/oracle_regret.py`. Artefatos:
> `results/models/oracle_regret/{report,ranking.csv,frontier.csv}`.

---

## 4. A hierarquia medida e a auditoria de composição

O vetor de atributos empregado nas soluções desta linha de investigação é
decomposto em cinco blocos rotulados. O **bloco A**, com vinte e quatro
descritores de luminância do bloco e do seu contexto hierárquico —
variância, gradientes, perfis de linha e de coluna, densidade de bordas,
contraste com o bloco-pai e com os irmãos —, é computável inteiramente a
partir dos pixels-fonte, antes de qualquer busca. O **bloco B**, com oito
atributos, carrega a vizinhança de particionamento causal já decidida
acima e à esquerda do nó. O **bloco C**, com quatro atributos, reúne
quantização e posição. O subconjunto `pixels24` é definido como o **bloco A
isolado**, e o subconjunto H9a como a **união A+B+C**, com trinta e seis
atributos — ou seja, `pixels24` é literalmente o bloco A do H9a, e não um
domínio de informação disjunto dele.

A Tabela 2 apresenta a hierarquia medida no crivo, do pior para o melhor
subconjunto, em dois pontos de operação de `cost_red` casado.

**Tabela 2** — Fração de perda de otimalidade (`reg_frac`, menor é melhor)
por subconjunto de atributos, a 25% e a 30% de redução de custo casada.
Seis sequências de validação e de teste reservado, 792.840 nós de decisão.

| subconjunto | atributos | `reg_frac` a 25% | `reg_frac` a 30% |
|---|---|---:|---:|
| variância | 1 (variância do bloco) | 0,0573 | 0,060 |
| ConvNeXt (alvo de perda de otimalidade) | 28,1 M parâmetros | 0,0219 | [não medido neste ponto] |
| ConvNeXt (entropia cruzada) | 28,1 M parâmetros | 0,0207 | [não medido neste ponto] |
| `pixels24` | 24 (bloco A) | 0,0121 | 0,015 |
| **H9a** | 36 (A+B+C) | **0,0036** | **0,006** |
| pontuação aleatória | — | [não medido neste ponto] | 0,612 |

> **Nota de procedência.** Dado de origem: `results/models/oracle_regret/frontier.csv`
> e `results/models/oracle_regret_convnext/frontier.csv`. Script:
> `src/scripts/partition_model/oracle_regret.py`. Fonte:
> `results/thesis/R1_dominio_pixels.md` §1.4. As duas variantes ConvNeXt não
> foram medidas no ponto de 30%, e a pontuação aleatória não foi medida no
> ponto de 25%; as células correspondentes são declaradas ausentes, e não
> estimadas.

Os retornos marginais entre subconjuntos são o resultado central desta
seção. Acrescentar os vinte e três descritores manuais restantes à
variância isolada — obtendo `pixels24` — compra **4,7×** de redução na
fração de perda de otimalidade. Substituir esses vinte e quatro descritores
manuais por 28,1 milhões de parâmetros convolucionais sobre os mesmos
pixels crus — obtendo o modelo substituto ConvNeXt — rende **0,6×**, ou
seja, **piora** o resultado. Acrescentar os doze atributos causais de
vizinhança, quantização e posição ao `pixels24` — obtendo H9a — compra
**3,4×** adicional. A Figura 1 apresenta esta comparação como fronteira
contínua ao longo de `cost_red`, e a Figura 2 apresenta os três retornos
marginais como estrutura ramificada.

A leitura correta desta hierarquia **não** é a de que o contexto de
taxa-distorção vence os pixels em geral, uma vez que os dois conjuntos
comparados na linha do H9a não são disjuntos: o enunciado correto e
verificável é duplo. No domínio de pixels, descritores manuais compactos de
luminância vencem uma rede convolucional profunda sobre pixels crus, por
cerca de 1,7×. E doze atributos causais de vizinhança, quantização e
posição, de custo praticamente nulo por já residirem na memória do
codificador, acrescentam 3,4× sobre eles. O que separa o melhor podador dos
demais não é, portanto, capacidade de representação sobre o bloco-fonte, e
sim o acesso a um contexto causal barato que a própria busca nativa já
consulta.

O modelo substituto convolucional **não estabelece cota superior alguma**
do domínio de pixels, e este não estabelecimento é, ele próprio, um
resultado desta investigação. Um modelo batido por outro de acesso
estritamente menor à informação não delimita limite superior algum: o
resultado enuncia algo sobre o treino realizado, e não sobre os pixels.
Duas medições adicionais sustentam esta leitura. Primeiro, retreinar a
mesma arquitetura contra o alvo correto — a perda de otimalidade, em vez de
entropia cruzada sobre rótulo duro — **piorou o modelo em toda a faixa**,
por fatores de 1,06× a 3,80×, no cenário mais favorável possível à hipótese
de que o objetivo de treino era a restrição. Segundo, dobrar a largura de
fusão do tronco convolucional, de 128 para 256 canais, altera a perda de
validação em apenas **0,16%**, o que estabelece que a capacidade de modelo
não é a restrição observada. O registro honesto do que este artigo pode
afirmar é, deste modo, o de uma **cota inferior** do domínio de pixels,
dada pelo `pixels24`, e de uma **cota superior genuína apenas no
oráculo** — a decisão de taxa-distorção ótima, de perda de otimalidade
nula —, que limita qualquer podador e não apenas os de pixels; a cota
superior do próprio domínio de pixels permanece **não medida**.

Uma quinta via do domínio de pixels foi fechada por medição, e é relatada
aqui por completude, sobre a hipótese de que o resíduo de uma predição
intra barata a partir dos vizinhos reconstruídos carregaria sinal de
predizibilidade adicional. A implementação inicial, que calculava o SATD do
próprio bloco-fonte em vez do resíduo de predição especificado, testou uma
hipótese diferente da formulada e reprovou sem informar sobre ela. A
hipótese correta, testada em separado sobre 13.000 superblocos de treino e
3.000 de validação, exigia entropia cruzada menor e área sob a curva maior
simultaneamente nos níveis de 16 e de 32 amostras. Em 32 amostras houve
sinal positivo consistente (entropia cruzada de 0,7422 contra 0,7657; área
sob a curva de 0,868 contra 0,851), mas sobre apenas 1.869 nós; em 16
amostras, o nível com 15.855 nós — oito vezes mais dados —, o efeito foi
nulo. O critério de decisão não foi atingido, e o resíduo positivo em 32
amostras fica registrado, sem ser perseguido, junto da observação de que o
nível de 64 amostras é **invisível a este crivo**, uma vez que a
disponibilidade do atributo ali é de 0,0%.

**Figura 1 — Fronteira de sobrecarga de taxa-distorção por redução de
custo, todas as soluções na mesma régua.**
*Legenda.* A Figura 1 apresenta `reg_frac` (eixo vertical, escala
logarítmica, menor é melhor) contra `cost_red` (eixo horizontal), para os
cinco subconjuntos de atributos da Tabela 2, com a pontuação aleatória como
piso, evidenciando que nenhuma via de pixels compete com o H9a na mesma
faixa de operação. *Tipo.* Dispersão com linha conectando os pontos de cada
subconjunto (uma curva por série). *Dado de origem.*
`results/models/oracle_regret/frontier.csv` e
`results/models/oracle_regret_convnext/frontier.csv`. *Roteiro.* Para cada
`pruner`, ordenar por `cost_red` e traçar `reg_frac_pct` em escala log;
anotar os dois pontos casados da Tabela 2 (25% e 30%) com marcador
diferenciado.
> **Nota de procedência.** Fonte: `results/thesis/R1_dominio_pixels.md`
> §1.4; especificação equivalente em `results/thesis/A2_TABELAS_E_FIGURAS.md`,
> Figura 2 (versão de ponto único a 25%, aqui estendida à curva completa).

**Figura 2 — Retornos marginais da hierarquia em quatro degraus.**
*Legenda.* A Figura 2 apresenta os quatro subconjuntos nomeados — variância,
`pixels24`, ConvNeXt (entropia cruzada) e H9a — em `reg_frac` a 25% de
`cost_red`, com os três retornos marginais anotados: variância→`pixels24`
(4,7×), `pixels24`→ConvNeXt (0,6×, ramo divergente e pior) e
`pixels24`→H9a (3,4×). *Tipo.* Gráfico de degraus (*step chart*) com um
ramo bifurcado a partir de `pixels24`: o ramo principal segue para H9a
(seta cheia), o ramo divergente segue para ConvNeXt (seta tracejada,
cor de alerta, apontando para cima por ser piora). *Dado de origem.* Os
mesmos dois arquivos da Figura 1. *Roteiro.* Extrair os quatro valores de
`reg_frac` a `cost_red`≈25% (Tabela 2); desenhar barras nas quatro posições
na ordem variância/ConvNeXt/`pixels24`/H9a; anotar as razões entre pares
adjacentes com as setas descritas.
> **Nota de procedência.** Fonte: `results/thesis/R1_dominio_pixels.md`
> §1.4 e §1.6 (derivação exata dos três fatores marginais). Sem CSV de
> retornos marginais dedicado; os três fatores são recalculados a partir da
> Tabela 2 (`0,0573/0,0121≈4,7×`; `0,0121/0,0207≈0,6×`;
> `0,0121/0,0036≈3,4×`), conforme já enunciado em prosa na fonte.

> **Nota de procedência da seção.** `results/thesis/M4_atributos_e_politica.md`
> §4.1 e §4.3 (decomposição do vetor de atributos);
> `results/thesis/R1_dominio_pixels.md` §1.4, §1.5, §1.6 e §1.7.

---

## 5. Validade do indicador substituto: a ablação controlada e a inversão de ordenação

A hierarquia da Seção 4 é obtida inteiramente fora do codificador, e a
seção anterior já declarou que o crivo **não adjudica**. Esta seção mede
diretamente até que ponto essa ressalva importa, por dois desenhos
independentes.

O primeiro é uma **ablação de atribuição de pequena escala**, executada
antes da construção do crivo, que compara o modelo estudante do domínio de
pixels (vinte e quatro atributos) contra um limiar de variância trivial
`exp(−var/1000)` e contra uma pontuação aleatória, sob política idêntica de
comprometimento com `PARTITION_NONE`, na sequência Jockey, com duas
imagens e quatro pontos de quantização. O resultado, restrito a esta
escala, é que o limiar de variância apresenta taxa BD **menor** que a do
modelo em todos os cinco níveis de aceleração medidos — de 0,171% contra
0,238% a 1,05× de aceleração, até 1,357% contra 1,895% a 1,55× —, ao passo
que a pontuação aleatória é largamente inferior aos dois (2,935% a 1,30×).
Este resultado, obtido num experimento de escala reduzida, **contradiz**
diretamente a hierarquia da Seção 4, na qual `pixels24` supera a variância
por 4,7×, e essa contradição permanece **sem árbitro** neste artigo: o
crivo de seis sequências mede o oposto do experimento de codificação de uma
sequência, e nenhum dos dois desenhos aqui apresentados substitui o outro
como resposta final sobre este par específico.

O segundo desenho é uma **ablação controlada pelo número de camadas de
passagem de mensagens**, aplicada a uma rede de grafos (do inglês *graph
neural network* – GNN) que decide o particionamento do superbloco de forma
conjunta, em vez de nó a nó. Com zero camadas, a rede degenera, por
construção, num perceptrone independente por nó, idêntico em espírito ao
H9a; com duas camadas, a estrutura passa a agregar informação da árvore
inteira. Todo o restante — atributos, dados, partição congelada, otimizador
— é mantido idêntico, de modo que qualquer diferença medida é atribuível
exclusivamente à estrutura.

Na simulação de oráculo, a estrutura conjunta supera de forma inequívoca o
nó independente. A redução de custo de busca a `SPLIT-lost` casado sobe de
42%, 52% e 52% (zero camadas) para **70,5%, 73,5% e 77,8%** (duas camadas)
nos pontos de risco de 0,5%, 1% e 2% — um ganho de cerca de **28 pontos
percentuais**. A Tabela 3 reúne este resultado ao lado do que se observa no
codificador real.

**Tabela 3** — A mesma comparação (rede de grafos contra perceptrone
independente/H9a) na simulação de oráculo e no codificador real,
sequência Jockey.

| painel | métrica | perceptrone / H9a | rede de grafos (2 camadas) |
|---|---|---|---|
| oráculo, `SPLIT-lost`≈0,5% | redução de custo | 42% | **70,5%** |
| oráculo, `SPLIT-lost`≈1% | redução de custo | 52% | **73,5%** |
| oráculo, `SPLIT-lost`≈2% | redução de custo | 52% | **77,8%** |
| codificador, ponto 1 | taxa BD / redução de tempo | **0,88%** / 25,1% | 1,50% / 27,6% |
| codificador, ponto 2 | taxa BD / redução de tempo | **0,75%** / 27,8% | 1,53% / 29,0% |
| codificador, ponto 3 | taxa BD / redução de tempo | **0,86%** / 30,3% | 1,59% / 30,4% |
| codificador, ponto 4 | taxa BD / redução de tempo | **0,94%** / 34,1% | 1,56% / 33,3% |

> **Nota de procedência.** Painel oráculo:
> `results/models/gnn_{L0,L2}/gate_oracle.csv`, script
> `src/scripts/partition_model/train_gnn.py` (opção `--causal`/`--feat-mode`
> não usada nesta linha). Painel codificador:
> `results/benchmark/gnn_frontier/frontier_Jockey.csv`, script
> `src/scripts/benchmark/gnn_frontier_bench.py`. Fonte:
> `results/thesis/R5_resultados_negativos.md` §5.2. A reprodução fiel das
> decisões da rede de grafos no codificador é obtida por reinjeção das suas
> probabilidades pelo mesmo ponto de inserção do H9a, sem que uma inferência
> de rede de grafos seja executada em C.

A leitura da Tabela 3 é a inversão de ordenação mais forte medida nesta
linha de investigação: o modelo que vence no oráculo por 28 pontos
percentuais **perde** no codificador real por um fator de aproximadamente
duas vezes em taxa BD, ao longo de toda a varredura de limiar. A fronteira
da rede de grafos no codificador é praticamente plana, entre 1,50% e 1,59%
de taxa BD, o que situa o limite na qualidade das decisões, e não na
calibração dos limiares. A causa exata desta derrota **não está
estabelecida** por este artigo — permanece pergunta aberta se decorre de
vazamento de informação na expressividade do grafo, de descasamento entre
o regime de treino e o de implantação, ou de outra ação da política —, e a
explicação inicialmente proposta, de que o custo seria concentrado em
poucas podas erradas caras, foi descartada por medição: as podas da rede de
grafos são baratas tanto por contagem (`SPLIT-lost` de 0,25%) quanto por
custo ponderado (`reg_frac` próximo de zero, a menor de todas as soluções
avaliadas no crivo).

A Figura 3 apresenta este resultado lado a lado, como o argumento central
desta seção: uma avaliação fora do codificador **filtra perdedores, mas não
coroa vencedores**, e uma margem relativa obtida fora dele é indício, e
nunca prova, de ordenação entre modelos competitivos.

**Figura 3 — O mesmo par de modelos no oráculo e no codificador, lado a
lado.**
*Legenda.* A Figura 3 apresenta, em dois painéis, a redução de custo de
busca medida na simulação de oráculo (esquerda) e a taxa BD contra a
redução de tempo medida no codificador real (direita) para o mesmo par de
modelos — a rede de grafos e o perceptrone H9a —, evidenciando que a
ordenação entre os dois se inverte de um painel para o outro. *Tipo.* Dois
painéis lado a lado: barras (oráculo) e dispersão com linha conectada
(codificador). *Eixos.* Esquerda — horizontal: modelo (perceptrone
independente; rede de grafos, 2 camadas); vertical: redução de custo (%) a
`SPLIT-lost`≈1%. Direita — horizontal: redução de tempo (%); vertical: taxa
BD (%); séries: rede de grafos, H9a. *Dado de origem.* Os quatro arquivos
citados na nota de procedência da Tabela 3. *Roteiro.* Ver
`results/thesis/A2_TABELAS_E_FIGURAS.md`, Figura 8, cujo roteiro completo é
reaproveitado sem alteração, restrito ao par (H9a, rede de grafos de duas
camadas) e ao painel de risco de 1%.
> **Nota de procedência.** Reaproveita integralmente a especificação da
> Figura 8 de `results/thesis/A2_TABELAS_E_FIGURAS.md`, redigida a partir
> de `results/thesis/R5_resultados_negativos.md` §5.2.

> **Nota de procedência da seção.** `results/thesis/R1_dominio_pixels.md`
> §1.3; `results/thesis/R5_resultados_negativos.md` §5.2;
> `results/thesis/M6_modelos_e_atribuicao.md` §6.4 e §6.10;
> `results/thesis/R7_ameacas_e_escopo.md` §7.1.1 e §7.3.3.

---

## 6. Classificação contra regressão: a formulação correta do problema

As soluções apresentadas nas seções anteriores formulam a poda como
**classificação** do rótulo de partição, podando por confiança sobre um
limiar aplicado à distribuição de saída. Uma reformulação alternativa,
igualmente legítima *a priori*, substitui esta formulação por uma
**regressão**: em vez de classificar o rótulo, o modelo regride
diretamente a perda de otimalidade de comprometer o nó com
`PARTITION_NONE`, e a poda é decidida por custo predito, em vez de
confiança predita. A motivação é que a fronteira entre taxa BD e tempo é
governada por *quanto* se perde ao podar, que é grandeza contínua, e um
preditor competente deste custo poderia, em princípio, podar exatamente
onde é barato.

O preditor de perda de otimalidade foi construído sobre os mesmos
trinta e seis atributos do H9a, com a mesma topologia de rede, isolando a
comparação à formulação do alvo. O obstáculo estrutural é a **inflação de
zeros**: a perda de otimalidade é nula para 59,5% dos nós de 64 amostras,
83,6% dos de 32 amostras e 98,1% dos de 16 amostras, e um regressor ingênuo
colapsa na distribuição *a priori*, prevendo perda de otimalidade próxima
de zero para quase todo nó. A Tabela 4 apresenta o resultado no terceiro
critério de decisão, que mede a redução de custo de busca a risco casado.

**Tabela 4** — Redução de custo de busca (%) a risco casado (`SPLIT-lost`),
classificação contra regressão, mesmos 36 atributos de entrada.

| formulação | 0,5% | 1% | 2% | piso de `SPLIT-lost` |
|---|---:|---:|---:|---:|
| regressão (ingênua) | 0,00 | 0,00 | 0,00 | 12,4% |
| regressão (balanceada) | 0,00 | 0,00 | 0,00 | ≈11% |
| variância (linha de base) | 4,46 | [não relatado no documento-fonte] | 6,35 | — |
| **classificador H9a** | **51,73** | **51,73** | **62,00** | **0,01%** |

> **Nota de procedência.** Dado de origem:
> `results/models/regret/{gate3_regret,gate3_h9a,gate3_var}.csv` e
> `results/models/regret_balanced/`. Script:
> `src/scripts/partition_model/train_regret.py`. Fonte:
> `results/thesis/R5_resultados_negativos.md` §5.1. O valor da variância no
> ponto de 1% de risco não consta do documento-fonte, que relata apenas os
> pontos de 0,5% e de 2%; a célula é declarada ausente, e não estimada.

A reponderação estatística da inflação de zeros foi implementada e medida,
e não assumida: uma variante balanceada, com pesos entre 8,21 e 45,56 sobre
os nós de perda de otimalidade não nula, aprendeu de fato a cauda da
distribuição, mas **não resgatou o ordenamento** — a redução de custo
permaneceu em 0,00% em todos os pontos de risco. Cerca de 11% dos nós cuja
decisão verdadeira é `SPLIT` têm perda de otimalidade aproximadamente nula,
e são, por isso, indistinguíveis dos nós seguros por qualquer regressão
sobre esta grandeza.

A conclusão de valor metodológico independe da magnitude dos números: a
decisão de poda é, no fundo, uma **classificação** entre seguro e inseguro,
e a **magnitude** da perda de otimalidade não é ordenável a partir de
atributos pré-busca baratos. O classificador se concentra exatamente na
fronteira de decisão que o ordenamento de poda exige, ao passo que o
regressor gasta capacidade em valores que não sobrevivem à predição. A
decisão de não levar esta reformulação ao codificador real não decorre de
implicação lógica alguma a partir do resultado no crivo — a Seção 5 já
estabeleceu que a simulação de oráculo pode inverter a ordenação entre
modelos competitivos —, e sim de uma **assimetria de custo experimental**:
o valor esperado da informação, dado um sinal já fraco na simulação, não
compensou o custo de integrar a formulação em C e de codificar por horas.

> **Nota de procedência.** `results/thesis/R5_resultados_negativos.md`
> §5.1; `results/thesis/M6_modelos_e_atribuicao.md` §6.5.

---

## 7. Confirmação no codificador sob política casada

As seções anteriores estabelecem validade e limites do crivo fora do
codificador. Esta seção apresenta a confirmação executada **dentro** do
codificador, sob política idêntica entre fontes de pontuação e a **tempo
casado** — o modo de comparação mais exigente, pois confronta a taxa BD dos
braços no mesmo valor de aceleração, sem interpolação alguma.

O experimento foi conduzido no conjunto de **validação** (FlowerPan e
Lips), cujo papel declarado no protocolo é a escolha de limiares
operacionais, e cobriu trinta e quatro pontos de operação, cento e
quarenta e quatro codificações reais e cerca de vinte e duas horas de
execução. Os três braços — o modelo estudante de trinta e seis atributos
(H9a), um limiar de variância e uma pontuação aleatória — rodam a mesma
política de comprometimento puro com `PARTITION_NONE`, no mesmo
codificador. A grade de limiares do braço do modelo foi mantida
**congelada**; apenas a grade do braço da variância foi estendida ao
extremo conservador, de modo que qualquer viés introduzido pela extensão
beneficiasse o adversário, e não a hipótese.

Na sequência **FlowerPan**, as faixas de aceleração dos dois braços se
sobrepõem, e o modelo domina em todos os pares medidos, sem interpolação:
0,095% de taxa BD contra 0,437% da variância a cerca de 1,15× (razão de
**4,6×**), e 0,638% contra 1,180% a cerca de 1,27× (razão de **1,85×**).
Contra a pontuação aleatória, as razões são muito maiores — 158× a cerca de
1,10× e 19× a cerca de 1,19×. Na sequência **Lips**, o resultado é de
natureza distinta e mais informativa: entre os limiares 0,99 e 0,97, a
variância salta de **1,006× para 3,563×** de aceleração e de 0,019% para
**6,58%** de taxa BD, sem ponto intermediário algum na grade, ao passo que
o modelo vive inteiramente dentro deste vão, de 1,074× a 1,283×. A ausência
de par casado nesta sequência não é limitação da medição: é um **resultado
sobre a pontuação da variância**, que atravessa em um único salto a região
em que o modelo opera continuamente.

A Tabela 5 resume os pontos casados e o resultado de taxa BD negativa
observado em ambas as sequências, em que o modelo economiza tempo com
ganho marginal de qualidade.

**Tabela 5** — Ablação de atribuição a tempo casado, dentro do codificador,
conjunto de validação.

| sequência | comparação | modelo | variância | razão |
|---|---|---:|---:|---:|
| FlowerPan | taxa BD a ≈1,15× | 0,095% | 0,437% | 4,6× |
| FlowerPan | taxa BD a ≈1,27× | 0,638% | 1,180% | 1,85× |
| Lips | taxa BD a τ=0,99→0,97 | [modelo permanece em 1,074×–1,283×] | 0,019% → 6,58% | sem par casado |
| FlowerPan | taxa BD negativa | **−0,015%** a 3,8% de redução de tempo | 0,000% a 0,2–1,7% de redução de tempo | — |
| Lips | taxa BD negativa | **−0,073%** a 6,9% de redução de tempo | 0,000% a 0,2–1,7% de redução de tempo | — |

> **Nota de procedência.** Dado de origem:
> `results/benchmark/e5_ablation/{FlowerPan,Lips}/{curve,runs}.csv`
> (não versionados). Scripts:
> `src/scripts/benchmark/ablation_attrib.py`, `analyze_ablation.py` e
> `run_e5_validation.sh`. Fonte:
> `results/thesis/R2_h9a.md` §2.6 (esta é a única seção de `R2_h9a.md`
> utilizada como fonte deste artigo).

O critério de decisão pré-registrado exigia dominância a tempo casado sobre
a variância em ao menos duas das três sequências de validação, e foi
atingido em **uma de duas** — a FlowerPan; na Lips não há par casado a
comparar, e a terceira sequência de validação, HoneyBee, foi cortada por
decisão de escopo declarada, por ser aquela em que a grade de limiares
original havia sido calibrada e, portanto, a menos independente das três.
**O critério estrito não foi atingido**, e este resultado é declarado como
tal. O que se sustenta, e não depende de sobreposição de faixas, é o
seguinte: sob política idêntica, a pontuação do modelo gradua
continuamente uma região implantável de baixa taxa BD, que a pontuação da
variância ou não alcança ou atravessa de um único salto; onde ambas
coexistem, o modelo custa de duas a cinco vezes menos taxa BD pelo mesmo
tempo; e, contra a pontuação aleatória, a atribuição é limpa e mensurável
nas duas sequências.

> **Nota de procedência da seção.**
> `results/thesis/R2_h9a.md` §2.6; `results/thesis/M6_modelos_e_atribuicao.md`
> §6.6 e §6.7.

---

## 8. Discussão, limitações declaradas e conclusão

O conjunto de evidências apresentado sustenta uma tese estreita e
verificável. A predizibilidade da decisão de particionamento intraquadro
não está na capacidade de representação sobre o bloco-fonte: uma rede
convolucional de 28,1 milhões de parâmetros sobre luminância crua perde
para um perceptrone de vinte e quatro descritores manuais extraídos da
mesma luminância, e doze atributos causais de vizinhança, quantização e
posição, de custo praticamente nulo, compram mais 3,4× sobre eles. O
indicador substituto usual — a margem relativa medida fora do codificador —
pode **inverter** a ordenação: um modelo estruturado que supera o modelo
implantado por vinte e oito pontos percentuais fora do codificador perde
para ele por um fator de dois dentro dele. Nenhuma das duas afirmações é
generalizável além do que foi medido, e as limitações declaradas a seguir
fixam esse alcance.

Seis lacunas permanecem abertas e devem acompanhar qualquer citação destes
resultados. **Primeira**, o critério estrito da ablação de atribuição
dentro do codificador foi atingido em apenas uma de duas sequências
comparáveis de validação; a Lips não oferece par casado, por a variância
atravessar de um salto a região em que o modelo opera. **Segunda**, o par
formado pelo subconjunto de vinte e quatro atributos de luminância e a
pontuação de variância permanece **sem árbitro** no codificador: o braço
aprendido da ablação a tempo casado é o modelo de trinta e seis atributos,
e não o `pixels24` isolado, de modo que a contradição entre o crivo de seis
sequências e o experimento de pequena escala da Seção 5 não é resolvida
por este artigo. **Terceira**, o custo de inferência do modelo substituto
convolucional nunca foi pago: toda medição por reinjeção de probabilidades
é uma cota superior que ignora esse custo, uma vez que nenhuma inferência
convolucional foi executada em C. **Quarta**, o resíduo positivo do
atributo de predizibilidade intra a partir dos vizinhos, em blocos de 32
amostras, não foi perseguido, e o nível de 64 amostras permanece invisível
ao critério aplicado, por disponibilidade nula do atributo. **Quinta**, a
grade de limiares do braço aprendido foi calibrada e congelada na
validação antes de qualquer medição no conjunto de teste reservado, e as
faixas de aceleração entre o modelo e a variância saíram disjuntas nas três
sequências de teste, de modo que a atribuição naquele conjunto repousa
sobre a comparação a política casada, e não sobre pares de tempo casado.
**Sexta**, a curva da pontuação de variância no crivo é amostrada em uma
grade esparsa, e todos os valores intermediários citados na Seção 4 são
interpolados através de um único vão entre 6,14% e 39,46% de redução de
custo.

Uma limitação adicional, de natureza distinta, concerne à publicação do
conjunto de dados. A ficha de publicação está redigida, mas a conversão
para o formato de arquivamento portátil e a atribuição de um identificador
digital de objeto **não foram executadas**; o compromisso de depósito é
declarado, e não o depósito em si. `[completar: confirmar licença de
redistribuição do conjunto de dados antes de qualquer publicação com
identificador digital de objeto]`.

Este artigo conclui que um número de aceleração, isoladamente, não é
evidência de aprendizado, e que a validade de um podador aprendido só se
sustenta contra fontes de pontuação alternativas, sob política idêntica, e
com o codificador — nunca a simulação — como árbitro final. A hierarquia
medida sobre o crivo ponderado por perda de otimalidade localiza a
informação relevante no contexto causal já disponível ao codificador, e
não na capacidade de representação sobre o bloco-fonte; a inversão de
ordenação medida entre a simulação e o codificador estabelece que essa
mesma simulação filtra candidatos inferiores, mas não coroa vencedores.
Trabalhos futuros devem, primeiro, arbitrar o par `pixels24` contra
variância dentro do codificador, com desenho equivalente ao da Seção 7;
segundo, investigar a causa da derrota do modelo estruturado, hoje
registrada apenas como pergunta aberta; e, terceiro, completar a
publicação do conjunto de dados com identificador digital de objeto e
licença confirmada.

---

## Referências

`[completar: preencher com citações bibliográficas reais; nenhuma
referência abaixo foi inventada — os marcadores indicam apenas o tipo de
trabalho que a posição deveria ocupar]`

[1] `[completar: especificação normativa do AV1 / documento de referência da Alliance for Open Media]`

[2] `[completar: artigo de origem do algoritmo de busca recursiva de particionamento por taxa-distorção em codificadores de vídeo baseados em blocos]`

[3] `[completar: trabalho relevante de poda aprendida do espaço de busca de particionamento em codecs de vídeo (HEVC/VVC/AV1), para contextualizar a literatura de referência]`

[4] `[completar: artigo original da métrica de Bjøntegaard (BD-rate)]`

[5] `[completar: artigo da arquitetura ConvNeXt]`

[6] `[completar: artigo de redes de grafos com passagem de mensagens sobre estruturas em árvore, por exemplo GraphSAGE ou GCN]`

[7] `[completar: artigo de calibração de redes neurais profundas / erro esperado de calibração, caso o artigo final discuta calibração]`

[8] `[completar: documentação ou artigo de referência do codificador libaom]`

---

## Conformidade

> **Apêndice interno de trabalho — NÃO faz parte da submissão.** Esta seção
> existe para auditoria interna do texto contra as retratações e lacunas
> registradas na tese de origem, e deve ser removida antes de qualquer envio.
> Ela é o único ponto deste arquivo em que decisões editoriais e a existência
> de outros trabalhos derivados da mesma tese são mencionadas.

Esta seção declara a aderência deste artigo às retratações e às lacunas
registradas em `results/thesis/A3_RETRATACOES_E_LACUNAS.md`, e lista os
marcadores `[completar: ...]` pendentes no corpo do texto.

### (a) Retratações verificadas

- **R1** (a saturação dos pixels na variância). O artigo não escreve que os
  pixels saturam na variância; apresenta a hierarquia medida no crivo
  (Tabela 2, Seção 4) e declara explicitamente a contradição com o
  experimento de pequena escala (Seção 5, lacuna L3).
- **R2** (o ConvNeXt como cota superior). O artigo escreve que o modelo
  substituto convolucional é instrumento de diagnóstico e tentativa
  documentada de estabelecer a cota; declara a cota inferior no `pixels24`
  e a cota superior genuína apenas no oráculo, com a cota do domínio de
  pixels não medida (Seção 4).
- **R3** (sobreajuste na seleção do modelo substituto). A alegação não é
  reproduzida: nenhuma afirmação de sobreajuste na escolha do ponto de
  verificação do modelo substituto aparece no texto.
- **R4** (composição disjunta entre H9a e `pixels24`). O artigo declara
  explicitamente que `pixels24` é o bloco A do H9a, que vinte e quatro dos
  trinta e seis atributos do H9a são descritores de luminância, e que o
  H9a não contém grandeza alguma de custo de taxa-distorção (Seção 4).
- **R5** (o ganho do contexto como margem competitiva). O artigo reporta os
  três retornos marginais (4,7×; 0,6×; 3,4×) como retorno marginal entre
  subconjuntos não disjuntos, nunca como margem competitiva entre domínios
  de informação distintos (Seção 4).
- **R12** (as conclusões H1–H6 sobre o sinal da luminância). O artigo não
  cita valor algum da cadeia H1–H6; menciona o defeito de luminância nula
  apenas como lição metodológica (Seção 2), sem invocar a cota aparente de
  13% a 18%.
- **R13** (o contexto necessário e suficiente). O artigo não afirma
  necessidade nem suficiência do contexto barato contra o podador nativo;
  nenhuma comparação contra *presets* nativos ou protocolo CTC é
  apresentada, por restrição de escopo declarada na Introdução.
- **R15** (o bloco D como proxy de resíduo de predição intra). O artigo
  registra, na Seção 4, que a implementação inicial testou uma estatística
  do bloco-fonte e não a hipótese de predizibilidade a partir dos vizinhos
  reconstruídos, e reporta o resultado do bloco D' corrigido, com o
  resíduo em 32 amostras e a invisibilidade em 64 amostras.
- **R16** (a regra de parada da regressão como implicação lógica). O
  artigo apresenta a decisão de não levar a regressão ao codificador como
  assimetria de custo experimental, nunca como implicação lógica do
  resultado no crivo (Seção 6).
- **R17** (a causa da derrota do modelo estruturado). O artigo declara
  explicitamente que a causa da derrota da rede de grafos no codificador
  permanece pergunta aberta, e não atribui a falha a "poucas podas erradas
  caras em RD" (Seção 5).
- **R22** (a subsunção do E5). Não aplicável a este artigo, que não discute
  a relação histórica entre o experimento de retreino do modelo substituto
  e a ablação a tempo casado dentro do codificador.
- **R23** (a comparabilidade direta entre definições de redução de tempo).
  O artigo não mistura as duas definições de redução de tempo do projeto;
  todas as reduções de tempo citadas (Seções 5 e 7) provêm de fontes
  únicas e internamente consistentes, sem comparação entre definições.

### (b) Lacunas declaradas

L2 (critério estrito do E5 atingido em uma de duas sequências), L3 (par
`pixels24`/variância sem árbitro no codificador), L5 (custo de inferência
do modelo substituto nunca pago), L9 (resíduo do bloco D' em 32 amostras e
invisibilidade em 64 amostras), L10 (grade de limiares congelada e faixas
disjuntas no conjunto de teste) e L12 (publicação do conjunto de dados não
executada) são declaradas explicitamente na Seção 8 deste artigo, cada uma
com o que de fato foi medido.

### (c) Marcadores `[completar: ...]` pendentes

1. Nomes dos autores, afiliação e financiamento (cabeçalho).
2. Confirmação do prazo, do número exato de páginas e de qualquer outro
   detalhe normativo do CFP da ICASSP 2027.
3. Licença de redistribuição do conjunto de dados, antes de qualquer
   publicação com identificador digital de objeto (Seção 8).
4. As oito entradas da lista de Referências, todas sem citação
   bibliográfica preenchida.

> **Nota de procedência geral.** Este artigo foi redigido exclusivamente a
> partir de `results/thesis/00_PLANO_capitulos.md` (regras editoriais §6),
> `M1_objeto_e_formulacao.md`, `M2_instrumentacao_e_dataset.md`,
> `M3_protocolo_avaliacao.md`, `M4_atributos_e_politica.md`,
> `M6_modelos_e_atribuicao.md`, `R1_dominio_pixels.md`,
> `R5_resultados_negativos.md`, `R2_h9a.md` (somente §2.6),
> `R7_ameacas_e_escopo.md`, `A2_TABELAS_E_FIGURAS.md` e
> `A3_RETRATACOES_E_LACUNAS.md`. Nenhum outro arquivo do projeto foi
> consultado, e nenhum artigo irmão derivado da mesma tese é citado ou
> pressuposto.
