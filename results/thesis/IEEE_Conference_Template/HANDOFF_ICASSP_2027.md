# HANDOFF — artigo ICASSP 2027 (atribuição de perda RD)

> **Arquivo temporário de retomada.** Registra onde o trabalho parou em 2026-08-20 e o que
> falta para escrever o artigo. Apagar quando o `.tex` estiver submetido.
> Commit de referência: `ec1d919` em `ml-partition-dev` (já empurrado).
> **Ler a §9 antes de escrever qualquer coisa** — é a explicação que passou no teste cego
> e a régua do rumo do artigo.

---

## 1. O que é este artigo

Segundo artigo da tese, para o **ICASSP 2027**. Não é artigo de solução: o objeto é
**representação e informação**, não aceleração de codificador.

**Chamada de trabalhos.** Até **4 páginas de conteúdo técnico incluindo figuras e
referências**, mais uma **5ª página opcional contendo apenas** referências,
agradecimentos de financiamento e declaração de conformidade ética. A tática é ocupar as 4
páginas com texto e figuras e empurrar as referências inteiras para a 5ª.

**Título escolhido (2026-08-21, definitivo):**
*Rate-Distortion Loss Attribution for AV1 Intra Partition Decisions*.

Variante longa, se a diagramação pedir corpo: *... Over Compact, Deep, and Causal
Representations*.

**Sem acrônimo algum — decidido em 2026-08-21, e isso substitui a decisão anterior.**
O LASCAS abre com acrônimo porque entrega um sistema; este artigo entrega uma medição, e
prefixo de acrônimo sinaliza solução. A primeira versão manteve **RPP — Representation Probe
for Partitioning** como nome do protocolo na Seção III, e isso foi retirado: o título fala em
*atribuição de perda de taxa-distorção* e o nome falava em *sondagem de representações*, duas
metáforas para o mesmo objeto, e a seção precisava abrir negando o que o próprio nome sugeria
(*"não é uma solução de codificador"*). O protocolo é chamado de **"the attribution protocol"**
ou **"the protocol"**, e o título carrega a identidade.

*Desvio consciente da Parte B, a registrar:* a B.1 quer o acrônimo no título da seção do
método, a B.2 tem a fórmula `This paper presents the <Nome> (<ACRÔNIMO>)` e a B.10 abre com
`This paper presented <ACRÔNIMO>`. As três estão escritas para artigo de solução; aqui a
contribuição é nomeada descritivamente.

*Nos artefatos o nome permanece* — `rpp_ladder.py`, `RPP_SUBSETS`, `results/models/rpp_ladder/`
e `oracle_regret_rpp/`. São nomes internos e não foram renomeados; não confundir com o texto.

*Títulos descartados, com o motivo (não reabrir):* `Causal Encoder State Outperforms Deep
Pixel Representations...` — **atribuía o número errado**: quem bate a ConvNeXt por 4,1× é o
bloco A, que é ele próprio luminância; o estado causal vale 1,60×, e o título fundia os dois
achados. `Luminance Is Not Sufficient for...` — correto, mas forma de proposição, e o ICASSP
publica sintagma nominal. `Optimality-Loss Attribution...` — *optimality loss* é vocabulário
nosso; *rate-distortion loss* é o termo que a literatura de predição de partição já usa para
a mesma coisa.

**Escopo decidido:** estritamente offline. **Nenhum** número de codificador (BD-BR ou
redução de tempo) entra neste artigo — evita sobreposição com o artigo LASCAS em revisão e
mantém o eixo no princípio (1).

**Regra vinculante:** o artigo **não cita o artigo LASCAS** (artigos irmãos da mesma tese
não se citam; `CLAUDE.md` B.15). Ele tem de se sustentar sozinho, e não pode descrever o
podador implantado como solução publicada.

## 2. Os quatro princípios que governam a escrita

1. O objeto científico é **information/representation**, não *speed-up*.
2. A métrica principal é **perda de otimalidade a redução de custo casada**, não
   acurácia/F1.
3. A ConvNeXt é **teste da hipótese** "pixels crus + capacidade bastam", não *baseline* de
   arquitetura.
4. O resultado mais importante é que **o estado causal do processo de codificação contém
   informação decisória ausente da luminância isoladamente** — e, após a medição, isso
   ficou mais preciso: é a **vizinhança de particionamento**, não o estado de quantização.

## 3. Estado: o que já está pronto

### 3.1 Experimentos — CONCLUÍDOS

| experimento | resultado | artefato |
|---|---|---|
| Escada RPP, 4 degraus × 3 sementes, receita única | A 0,0053 · A+B 0,0033 · A+C 0,0064 · A+B+C 0,0040 (a 25%) | `results/models/rpp_ladder/` |
| Grade densa de τ da variância (26 pontos) | 0,0374 medido a 25%; vão fechado | `oracle_regret.py`, `VARIANCE_TAUS` |
| ConvNeXt-CE, corpus correto (α=0, fusão 128) | val 0,895092, época 5 | `results/models/surrogate_ce_h9/` |
| ConvNeXt-CE, controle de capacidade (fusão 256) | val 0,903984 — **piora 1,0%** | `results/models/surrogate_ce_h9_f256/` |
| Fronteira única, 26 pernas, 3.808.703 nós | tabela completa | `results/models/oracle_regret_rpp/frontier.csv` |

### 3.2 Documentação — CONCLUÍDA

- `docs/RESULTADOS_rpp_escada_informacional.md` — resultados, procedência, reprodução, 6
  limitações.
- `docs/RESULTADOS_auditoria_convnext_corpus.md` — auditoria do corpus do braço profundo.

### 3.3 Correções na tese — CONCLUÍDAS

Propagadas em `M4`, `M6`, `R1`, `R2`, `R4`, `R5`, `R6`, `R7`, `A1`, `A2`, `T_RESULTADOS` e
nos docs operacionais (`ANDAMENTO`, `SINTESE`, `INVENTARIO`, `DECISOES_escopo`,
`RESULTADOS_auditoria_dominio_pixels`, `RESULTADOS_convnext_regret`).

## 4. Os números do artigo — fonte única

**Todos** de `results/models/oracle_regret_rpp/frontier.csv`. Vara: 6 sequências held-out
(HoneyBee, FlowerPan, Lips, Jockey, RaceNight, RiverBank), **226.447 superblocos**,
**3.808.703 nós de decisão**, custo RD total 1,28×10¹⁴.

`reg_frac` em **por cento do custo RD total**; menor é melhor. Degraus = média de 3 sementes.

| braço | 5% | 10% | 15% | 20% | 25% | 30% |
|---|--:|--:|--:|--:|--:|--:|
| aleatório | 0,0980 | 0,1963 | 0,2989 | 0,4066 | 0,5166 | 0,6342 |
| variância isolada | 0,0007 | 0,0043 | 0,0127 | 0,0250 | 0,0374 | 0,0555 |
| ConvNeXt-CE, 28,1 M | 0,0029 | 0,0065 | 0,0105 | 0,0164 | 0,0272 | 0,0442 |
| ConvNeXt-CE, fusão 256 | 0,0036 | 0,0075 | 0,0122 | 0,0194 | 0,0294 | 0,0444 |
| ConvNeXt, α = 3 | 0,0033 | 0,0054 | 0,0080 | 0,0140 | 0,0219 | 0,0379 |
| A (24 col) | 0,0002 | 0,0005 | 0,0013 | 0,0028 | 0,0053 | 0,0091 |
| A+B (32 col) | 0,0001 | 0,0002 | 0,0005 | 0,0015 | 0,0033 | 0,0069 |
| A+C (28 col) | 0,0002 | 0,0005 | 0,0012 | 0,0027 | 0,0064 | 0,0122 |
| A+B+C (36 col) | 0,0000 | 0,0002 | 0,0006 | 0,0018 | 0,0040 | 0,0078 |

Por semente a 25% (é o que sustenta as separações):

| degrau | s0 | s1 | s2 |
|---|--:|--:|--:|
| A | 0,0057 | 0,0049 | 0,0054 |
| A+B | 0,0034 | 0,0030 | 0,0036 |
| A+C | 0,0090 | 0,0056 | 0,0045 |
| A+B+C | 0,0040 | 0,0040 | 0,0040 |

**Razões a citar (25%):** variância → A = **7,0×** · A → A+B = **1,60×** · ConvNeXt(α3) → A
= **4,1×** · parâmetros ConvNeXt / MLP = **2.062×** (28.128.638 contra 13.641 nas três
redes de 36 colunas) · fusão 256 contra 128 = **1,08× pior**.

**Separações robustas (suporte disjunto, 3×3):** A+B vence A em todas as nove comparações
(pior A+B 0,0036 < melhor A 0,0049); A+B vence A+B+C em todas as nove (0,0036 < 0,0040).
**Não robusta:** A+C contra A — distribuições sobrepostas, dispersão dobrada. Enunciar como
"C não acrescenta", nunca "C prejudica quando isolado".

### 4.1 Como a grandeza entra na tabela — decidido em 2026-08-21

**A normalização NÃO será refeita.** `reg_frac_pct = 100 · Σ reg_abs / Σ none_rd` fica como
está.

Existe um denominador melhor e ele é barato: o custo RD ótimo da vara,
`Σ_superbloco [none_rd(raiz) − reg_abs(raiz)]`, que **não** conta a mesma área três vezes e
faria a coluna ler como "acréscimo percentual sobre a codificação RD-ótima". Uma passagem só
de denominador custa 20–40 min (lê os 24 pkls held-out, ~12 GB, sem features e sem pontuar
modelo) e o CSV já traz `reg_abs`, então bastaria reescalar — a fronteira inteira, que levou
2 h 35 min, não precisaria rodar de novo.

**Foi descartado assim mesmo**, e o motivo é de propagação, não de compute: `reg_frac` é
citado em **20 arquivos** de `results/thesis/` e `docs/`, e vários deles
(`RESULTADOS_oraculo_regret`, `approachB`, `B1`, `B2`, `B4`) reportam `reg_frac` de execuções
**diferentes**, as de 792.840 nós, que essa passagem não tocaria. O resultado seria duas
normalizações convivendo sob o mesmo nome de grandeza — exatamente a classe de erro que já
custou caro uma vez (ver §7). Reabrir isso exige refazer o denominador de *todas* as
execuções citadas, não só a do ICASSP.

**O que se faz em vez disso**, e resolve a confusão onde ela nasce, que é no texto:

1. Declarar o denominador **explicitamente**, uma vez, antes do primeiro número (regra B.0).
2. **Escalar a coluna da tabela para `10⁻³ %`**: a variância lê **37,4** e A+B lê **3,3**, sem
   parede de zeros à esquerda. O dado não muda.
3. Deixar a razão visível na tabela ou na legenda — **11,3×** entre variância e A+B a 25%.

O título sobrevive a essa leitura porque *Attribution* é um uso **ordinal**: o artigo atribui
a perda às representações, não afirma magnitude absoluta.

## 5. Estrutura acordada do artigo

- **I. Introduction** — M1 a M8 do `CLAUDE.md` B.4. A lacuna (M6): os trabalhos existentes
  propõem uma política e reportam velocidade; nenhum separa a qualidade da fonte de
  pontuação da agressividade da política, nem mede onde reside a informação preditiva.
- **II. The Partition Decision in AV1** — dez formas, ordem fixa de avaliação e, o ponto
  próprio deste artigo, **que informação já foi paga em cada instante**. Fecha com
  delimitação de escopo (intra, `cpu-used=0`, ação NONE-commit isolada).
- **III. Proposed Attribution Protocol**
  - III-A as representações, declaradas por **acesso à informação**, não por arquitetura.
  - III-B o protocolo: política fixa, varredura de τ, modelo de custo `candidatos × n²`,
    `regret_abs(n) = none_rdcost(n) − RD_subtree(n) ≥ 0` normalizado — equação numerada.
    **Redigir a partir da §9**, que é o esqueleto pronto desta subseção.
    Declarar que **não é tempo de parede**.
  - III-C dataset e partição, com declaração antivazamento.
  - III-D **a justiça do braço profundo** (princípio 3): corpus correto, sem vazamento,
    capacidade dobrada e os dois objetivos. Aqui entra a inversão do α.
- **IV. Results and Discussion** — tabela da fronteira; escada com as razões; a
  decomposição B contra C; limitações como resultado.
- **V. Conclusions** — fórmula B.10, sem número inédito.

**Decisões de tratamento já tomadas:**
- A inversão do objetivo (α=3 melhor que α=0) **NÃO entra** no artigo — revisto em 22/08,
  substitui a decisão anterior. Contratempo de processo não é achado. A III-D diz apenas
  que os dois objetivos foram treinados sob corpus, escalonamento e critério idênticos,
  que o sensível a custo vence de 10% em diante, e que o braço profundo é lido no melhor
  dos dois. Pelo mesmo motivo saiu a auditoria do checkpoint contaminado: no lugar dela, a
  afirmação positiva de que o braço foi treinado no mesmo corpus e sob a mesma partição
  dos degraus. **A retratação permanece na tese**, em `M6` e `R7`, onde ela pertence.
- Os **quatro** degraus entram na tabela, incluindo que **A+B domina A+B+C**.

## 6. O que falta — a fazer amanhã

1. ~~**Escrever o `.tex`**~~ — **FEITO em 2026-08-21.** `PAPER_ICASSP_2027_RD_ATTRIBUTION.tex`,
   compilado no contêiner `latex_build` (monta `results/` em `/work`). 5 páginas: as 4
   técnicas fecham na Conclusão e a 5ª abre no Acknowledgment, contendo só agradecimento,
   e as 26 referências, como a CFP permite. A declaração de conformidade ética foi
   **removida por decisão do autor em 21/08** — a CFP a admite na 5ª página, não a exige. Zero `Overfull`. Balanceamento da última
   página por `IEEEtriggeratref{18}` — **reconferir se a lista de referências mudar**.
   As figuras estão reservadas como `ramebox` com o propósito escrito dentro, marcadas
   `% TODO`; Fig. 1 com **2,8 cm** e Fig. 2 com **4,4 cm** de altura reservada — a folga
   de texto acabou, então uma figura mais alta que a reserva empurra a Conclusão para a 5ª
   página e viola a CFP.
2. **Gerar as figuras por último**, no contêiner `av1_bench` com `build/venv-ml`, em inglês
   e paleta acadêmica sóbria. Duas previstas:
   - Fig. 1 — linha do tempo de disponibilidade de informação dentro do nó. **Desenho
     novo**, não reaproveitar a Fig. 2 do LASCAS.
   - Fig. 2 — curvas de perda de otimalidade × redução de custo casada, eixo y logarítmico,
     com barra de amplitude das sementes nos degraus. É a figura que carrega o artigo.
3. **Checklist da Parte C** do `CLAUDE.md`, compilar e conferir a paginação (4 técnicas + 5ª
   só de referências e financiamento).
4. **Commit + push.**

## 7. Armadilhas a não repetir

- **Sempre Python no contêiner**: `docker exec av1_bench bash -lc 'cd /workspace && build/venv-ml/bin/python ...'`. O caminho é `build/venv-ml`, **não** `venv-ml`. Não há Python no PATH do Windows.
- **Não concatenar** `oracle_regret/frontier.csv` com `oracle_regret_convnext/frontier.csv`:
  cobrem varas de tamanhos diferentes (792.840 contra 3.808.703 nós). Foi assim que a tese
  acabou misturando duas execuções num parágrafo só.
- `reg_frac` é **percentagem**, não fração. 0,0374 é 0,0374% do custo RD total.
- **Coluna ≠ descritor conceitual.** O bloco A tem 24 colunas, das quais 21 de luminância; a
  posição é um conceito e duas colunas. Foi essa ambiguidade que gerou o off-by-one.
- 792.840 nós continua **correto** no portão de predizibilidade do H9d e nos experimentos
  B1/B4 — não "corrigir" aqueles.

## 8. Ressalvas obrigatórias no texto do artigo

Pela regra B.0 (a limitação vem **antes** do número):

- O crivo **não adjudica**: no único par com chão de codificador limpo ele diverge do
  encoder. É instrumento de atribuição de informação, não preditor de codificador.
- O denominador soma custos **mutuamente excludentes** (os três níveis recobrem a mesma
  área), o que torna a grandeza sistematicamente menor que qualquer perda observável no
  codificador. Legítima apenas em regime **ordinal**.
- Rótulos são o ótimo RDO a `cpu-used=0`.
- Três sementes dão **amplitude**, não intervalo de confiança.
- α = 0 e α = 3 são dois pontos; a curva em α não foi varrida.
- A superioridade de A+B sobre A+B+C **não foi verificada no codificador**.

---

## 9. Explicação de referência — o artigo para quem chega pela primeira vez

> **Por que isto está aqui.** Esta é a versão que sobreviveu ao teste cego: um leitor que não
> conhece o projeto entende o propósito, o que foi medido, de onde vêm os números e por que
> eles não são BD-BR. É o esqueleto da **Seção III-B** e a régua para não derivar do rumo.
> Reler antes de escrever qualquer parágrafo de método. Está em português porque é material
> de trabalho; o artigo é em inglês.

### 9.1 O que o artigo quer descobrir

Quando o AV1 codifica um quadro intra, ele decide como recortar cada superbloco de 64×64. Em
cada nó dessa árvore ele testa até dez formas de partição e desce recursivamente. A
`cpu-used=0` ele testa tudo, e por isso o recorte que ele escolhe é, por definição, o ótimo
em taxa-distorção.

A literatura de aceleração ataca isso do mesmo jeito há anos: treina um modelo que olha o
bloco e decide parar cedo. Cada trabalho propõe um conjunto de atributos, uma arquitetura e
um limiar, e reporta quanto tempo economizou. O que ninguém separou é **qual parte do ganho
vem de o modelo enxergar melhor, e qual parte vem simplesmente de ele podar mais**. São
coisas diferentes: dá para parecer melhor apenas sendo mais agressivo, ao custo de qualidade
que a métrica de tempo não mostra.

Este artigo não propõe acelerador nenhum. Ele pergunta: **que informação um modelo precisa
ter, na frente dele, para decidir bem?** Pixels bastam? Capacidade de representação compensa
a falta de informação? Ou existe informação que simplesmente não está na imagem?

### 9.2 O que foi medido

Uma única ação, isolada: no nó, comprometer-se com NONE — não partir — e não descer mais.

Isso tem duas consequências mensuráveis. A primeira é **trabalho poupado**: todas as formas
que não foram avaliadas naquele nó e em toda a subárvore abaixo dele. A segunda é
**qualidade perdida**: se o ótimo era partir, o custo RD de codificar aquela região sobe.

As duas saem do **mesmo registro**. O codificador instrumentado, rodando a busca completa uma
vez, já anotou o custo RD de cada nó e de cada subárvore. A perda de um nó é
`custo_RD(NONE no nó) − custo_RD(subárvore ótima)`, que é zero quando NONE já era o ótimo e
positiva quando não era. **Nada é recodificado**: o estudo inteiro é um *replay*
contrafactual sobre o log dessa busca exaustiva.

### 9.3 De onde vem o "25%"

Cada representação tem um limiar de confiança. Baixando o limiar, ela poda mais: economiza
mais trabalho e perde mais qualidade. Cada representação, portanto, não é um ponto — é uma
**curva**.

Comparar duas curvas em pontos diferentes não diz nada. Então o protocolo varre o limiar de
cada representação e lê todas elas **no mesmo ponto de trabalho poupado**. Vinte e cinco por
cento é esse ponto de leitura: a configuração em que um quarto das avaliações de forma deixou
de ser feita. A pergunta que a tabela responde é literalmente *"poupando exatamente o mesmo
esforço, qual representação estraga menos a decisão?"*

Nesse ponto, a variância do bloco — o atributo clássico — desperdiça **37,4** unidades da
escala da tabela; as trinta e duas colunas compactas com contexto causal desperdiçam **3,3**.
Onze vezes menos, com o mesmo esforço poupado. É esse contraste, e não o valor isolado, que
carrega o artigo.

### 9.4 Por que isso não é BD-BR × TS — o parágrafo que vem antes da primeira tabela

A forma é a mesma — custo de qualidade contra economia de esforço — mas as duas grandezas são
de natureza diferente, e isso tem de ser dito de uma vez.

*Time savings* é relógio: mede-se codificando duas vezes e comparando durações, com toda a
variação de máquina que isso traz. O eixo daqui é **contado, não cronometrado**: é o número de
candidatos de forma que a busca deixou de avaliar, ponderado pela área. Não há segunda
codificação, não há ruído de medição, e o número é exatamente reprodutível.

*BD-BR* é diferença de taxa a qualidade igual, integrada sobre quatro pontos de quantização, e
exige codificar tudo de novo com o podador ligado. O eixo daqui é o **acréscimo do custo
Lagrangeano que a própria busca já calculou**, lido do registro. Ele diz quanta otimalidade a
decisão jogou fora, não quantos bits o arquivo final ganhou.

A consequência pragmática, sem rodeio: **estes números não convertem em BD-BR e não devem ser
lidos como tal**. São sistematicamente menores, por dois motivos — o denominador agrega os
custos dos três níveis da árvore, que recobrem a mesma área da imagem, e a medição cobre uma
única ação de poda, não uma cadeia completa de codificação. Servem para **ordenar**
representações sob esforço casado, que é a pergunta do artigo, e não para prever o que um
codificador entregaria.

Essa limitação é a contrapartida do que o método dá em troca: comparar dezenas de
representações sobre 3,8 milhões de nós reais, com a mesma política e o mesmo esforço, sem
recodificar uma única vez. Um estudo assim é impraticável com BD-BR.

---

## 10. Estado da Parte C do checklist (verificado em 2026-08-21)

**Cumprido e verificado por compilação:** limite de páginas (4 técnicas + 5ª só de
referências e agradecimento); nenhum `Overfull`; nenhuma referência indefinida;
`\label` sempre depois de `\caption`; legenda de figura abaixo e de tabela acima; nenhum
pacote de margem ou fonte; equações dentro da coluna e sem `eqnarray`; citações agrupadas em
`\cite{a,b,c}`; sem matemática no título, no *abstract* e nos *index terms*; AV1 redefinido no corpo; colunas da última página equalizadas à mão; `Fig. N`, `Table N` e
`\eqref` nas chamadas. *Abstract* com **248 palavras**, dentro da faixa de 150–250 da B.2.

**Pendente, e só resolvível quando as figuras existirem:** figuras vetoriais para *line art*
com fontes Type 1 embutidas; rótulos de eixo com palavras e unidade entre parênteses, a 8 pt.
Ao trocar cada `\framebox` pelo `\includegraphics`, **recompilar e reconferir a paginação** —
o texto foi ajustado com folga de poucas linhas, e uma figura mais alta que a reserva empurra
a Conclusão para a 5ª página, o que viola a CFP.

---

## 11. Desvios conscientes da Parte B (não "corrigir" sem falar com o autor)

1. **Sem acrônimo** — ver §1. Afeta B.1 (acrônimo no título da seção do método), B.2
   movimento 3 e B.10 movimento 1.
2. **M1 sem macrocontexto de demanda** — decidido pelo autor em 2026-08-21. A B.4 M1 pede
   abertura por demanda de vídeo/tráfego com fonte citável (Cisco, Statista). O artigo abre
   **conceituando codificação de vídeo e perda de eficiência de codificação**, porque o objeto
   aqui é a perda de otimalidade e não a aceleração; um parágrafo de tráfego de Internet não
   sustenta nada do que vem depois. A referência `statista` foi **removida da bibliografia**
   (referência não citada não pode ficar na lista). A bibliografia tem hoje **26** entradas,
   contando a de aprendizado sensível a custo acrescentada em 21/08. O parágrafo fecha em *"That degradation, and not encoding time, is the quantity
   this work measures"*, que faz o serviço de enquadramento que o M1 pedia.
3. **Movimento 5 do abstract sem taxa BD nem redução de tempo** — consequência do escopo
   offline (§1). No lugar entram as razões a custo casado.

---

## 12. Correção de fato sobre a estrutura de particionamento (21/08)

O artigo dizia que o AV1 divide o quadro em superblocos de 64×64. **Está errado, e a
correção tem consequência de escopo.**

**No texto do artigo o fato é atribuído a `han2021` (Han et al., "A technical overview of
AV1"), por decisão do autor**, sem citar o limiar de 720p nem o código. A verificação
interna, que sustenta a decisão de escopo mas não aparece no artigo, foi esta:
`av1_select_sb_size()` em `src/aom/av1/encoder/encoder_utils.c` —
com `--usage=2` (All-Intra), `cpu-used=0`, 4K, sem superres nem resize e sem `--sb-size`
explícito no `build_dataset.py`, o fluxo cai no `return BLOCK_128X128`. O
**superbloco era 128×128**. O dataset confirma: `label_histogram.csv` só tem
`block_dim` 8, 16, 32 e 64.

Portanto:

- a árvore do AV1 desce até **4×4**, não até 8×8;
- a instrumentação engancha nos **quatro quadrantes de 64×64** de cada superbloco, e a
  **decisão do nó raiz de 128×128 está fora do escopo** — isso agora está declarado no
  fecho da Seção II;
- 8×8 é folha **empírica**, não estrutural: nenhum rótulo dividido entre 16,91 milhões de
  amostras, de modo que nós de 4×4 não ocorrem neste corpus;
- os 226.447 não são superblocos, são **unidades de 64 amostras** — o texto foi corrigido;
- a entrada da ConvNeXt é a luminância da **unidade de 64×64**, não do superbloco.

**Justificativa do recorte em 4K, acrescentada na Seção III-C** para não ser rebatida na
banca: o custo da busca cresce com o número de nós, então é nesta resolução que a busca
exaustiva é mais proibitiva; e é o regime em que o libaom seleciona o superbloco de
128×128 e, portanto, a árvore mais profunda — de modo que **misturar resoluções seria
mediar sobre duas estruturas de árvore distintas**. A contrapartida entra como quarta
limitação nos Resultados: nada aqui estabelece se a vizinhança causal pesa o mesmo onde o
superbloco é 64×64 e a árvore tem um nível a menos.

**A tese não repete o erro** — `M2` e `M3` descrevem o corpus sem atribuir tamanho de
superbloco. Mas vale conferir se algum documento afirma "superbloco de 64×64" antes da
defesa.

---

## 13. Revisão da Seção III-B pelo olhar do leitor de codificação (22/08)

Cinco mudanças aplicadas, todas com o mesmo motivo: o que um leitor do ViTech tropeçaria.

1. **Nove contra dez — era o defeito atacável.** A Seção II diz dez formas; o contador de
   custo usa nove. O motivo estava só no código (`simulate_pruning.py`, `CANDS`): o
   **`PARTITION_SPLIT` fica de fora**, porque o custo dele é a recursão nos filhos, já
   cobrada lá. Contá-lo no nó seria contar duas vezes. Agora está dito no texto. **Não
   remover essa oração.**
2. **A equação do custo de busca foi cortada.** `c(n) = k·n²` é regra de contagem com dois
   valores; virou uma frase, que já embute o item 1. Havia inversão de importância: `Δ` é o
   eixo de todas as figuras e recebia prosa, enquanto o bookkeeping recebia número. Restam
   **duas** equações, a perda por nó e a perda agregada — as duas se pagam.
3. **A ressalva do denominador subiu** para logo depois da equação da perda. Antes vinha
   dois parágrafos atrasada, e o leitor formava a objeção do recobrimento triplo antes de
   encontrar a resposta.
4. **Nomes de sintaxe do AV1** — `PARTITION_NONE` e `PARTITION_SPLIT` — no lugar de
   "undivided shape" e "quaternary split". Para quem lê libaom, é mais rápido.
5. **Terceira ressalva, nova e antes ausente:** o somatório percorre os **quatro pontos de
   quantização**, cujos multiplicadores de Lagrange diferem, de modo que o agregado pende
   para o mais grosseiro, onde os custos RD são maiores. Não invalida o regime ordinal, mas
   é pergunta certa de banca de codificação. **Não foi quantificado** — exigiria uma
   passagem sobre os pkls somando `none_rd` por CQ.

Menores: "optimal subtree" virou linguagem de partição; `Δ = 25%` glosado como *"one
quarter of the search work skipped"* e não "of the shape evaluations", que era impreciso
por Δ ser ponderado por área.

---

## 14. Vocabulário: "rung" foi eliminado (22/08)

O termo vinha da *escada informacional* interna, mas **o artigo nunca estabelece a metáfora**
— nunca diz "ladder" —, de modo que o leitor encontrava "rung A" sem referente. Pior, era
uma segunda taxonomia para algo que o artigo já nomeia: *representation* é o título da III-A
e o sujeito do protocolo e dos resultados.

Esquema adotado, a manter:

- **classe** → *representation* (ou *tabular representation*, quando o contraste é com o
  braço profundo);
- **indivíduo** → o rótulo nu: **A**, **A+B**, **A+C**, **A+B+C**;
- **artefato treinado** → *networks* / *model*, e só onde parâmetros ou treino são o assunto;
- **procedimento de treino** → *training strategy*, sempre. Havia três nomes para isto —
  "recipe", "training procedure" e "training strategy" —; ficou só o último, que é o que a
  literatura IEEE de codificação usa.

**Não trocar por "model"**: a tese do artigo é que a diferença não está no modelo e sim na
informação, e chamar A e A+B de *models* empurra o leitor para a leitura arquitetural que o
artigo quer negar. Nos artefatos de código a escada permanece (`rpp_ladder.py`), ver §1.
