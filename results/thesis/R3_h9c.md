# 3. H9c — refinamento pós-NONE, o fator de confusão e o resultado limpo

Esta seção apresenta a terceira solução investigada nesta tese — o podador H9c, um
refinamento que age **depois** da avaliação de `PARTITION_NONE` e que utiliza, como
informação de entrada, o custo de taxa-distorção real desta avaliação — e relata,
com o detalhamento que o caso exige, o episódio de autocorreção experimental que a
caracterização desta solução impôs. São apresentados, nesta ordem, o projeto do
refinamento e o seu ponto de inserção no fluxo de controle do codificador, o
critério de decisão offline atingido com folga e o contraste entre este desempenho
e o que o codificador viria a medir, o fator de confusão que contaminou as
primeiras medições em codificação real e a sua correção quantificada, o resultado
limpo da substituição direta nas oito sequências das condições comuns de teste
(do inglês *Common Test Conditions* – CTC), a decomposição que separa os ganhos
empilhados e mede a interação entre o H9a e o H9c, a verificação combinada e, por
fim, o veredito. O episódio de retratação é relatado por inteiro, pois o valor
metodológico que ele carrega para esta tese é da mesma ordem do resultado positivo
que o sucedeu.

---

## 3.1 O projeto — poda com o custo de taxa-distorção já pago

O H9c foi projetado a partir de uma pergunta direta, deixada em aberto pela
Solução 2: qual é o desempenho alcançável por um podador que decide **depois** de
o codificador já ter pago a avaliação de `PARTITION_NONE`, dispondo, então, da
taxa, da distorção e do custo de taxa-distorção reais desta hipótese. O vetor de
atributos do H9c é o vetor de trinta e seis atributos do H9a acrescido do bloco E,
que traz exatamente estas três grandezas, totalizando **trinta e nove atributos**;
a arquitetura é um perceptrone de múltiplas camadas leve, de dimensões
39→64→32→3, treinado por tamanho de bloco como todos os modelos desta família.

O ponto de inserção no fluxo de controle é a chamada de poda
`av1_prune_after_none`, que executa dentro de `av1_rd_pick_partition`
imediatamente após `none_partition_search()`, e a ação do H9c é **binária**:
decidir se a busca continua pelos candidatos restantes — a divisão quadrada, as
duas retangulares, as quatro AB e as duas 4-way — ou se o nó é encerrado ali,
reaproveitando o mecanismo `av1_disable_all_splits` já existente no codificador.
Deste modo, o H9c situa-se em posição distinta do H9a na dimensão *quando* do
espaço de projeto descrito na Seção 1.4, mas ocupa **a mesma posição na dimensão
*o quê***, pois ambos os podadores formulam, no fundo, a mesma pergunta sobre o
nó: se é possível parar ali. Esta coincidência de ação é o que torna previsíveis os
resultados de composição relatados nas Seções 3.5 e 3.6.

A solução é ativada por uma guarda de compilação e ambiente,
`AV1_STUDENT_H9C_ENABLE`, desligada por padrão, de modo que o codificador com a
guarda inativa produz fluxo de bits byte a byte idêntico ao da linha de base. Esta
decisão de projeto — que se mostraria decisiva no episódio da Seção 3.3, por
contraste com o podador H9a, que **não** possui guarda equivalente — é a mesma
política de inércia adotada para todas as soluções integradas em C nesta tese.

> **Procedência.** `docs/SINTESE_resultados_metodologia.md` §5 (motivação, projeto,
> bloco E e arquitetura); `docs/ANDAMENTO_tese.md` §7 (vetor de 39 atributos e
> descrição do ponto de inserção); `docs/INVENTARIO_solucoes.md` §4 (família C,
> 39 atributos A+B+C+E, ação binária);
> `src/aom/av1/encoder/partition_strategy.c:2269` (`av1_prune_after_none`) e `:2137`
> (`student_h9c_enabled`). Plano de origem:
> `docs/superpowers/plans/2026-07-15-h9c-teto-rd-pos-none.md`.

---

## 3.2 O critério de decisão offline, atingido com folga

O H9c foi submetido à mesma cascata de critérios de decisão descrita na Seção 3.7
do Capítulo de Metodologia, e o resultado do critério de validação por simulação
de oráculo foi o melhor obtido por qualquer solução desta tese. Sobre o conjunto
de validação, o H9c atinge **61,2% de redução de custo a apenas 0,20% de perda de
SPLIT**, ou seja, cinco vezes menos risco do que a cota superior de 1% admitida
pelo protocolo, superando o próprio H9a no mesmo critério, que alcança de 55% a
58% de redução de custo a perda de SPLIT de até 1%. O critério de integração em C
também foi atingido: os atributos 0 a 35 são bit-exatos contra a extração de
referência, os atributos 36 a 38 do bloco E são verificados por construção — pois
o código em C aplica a transformação logarítmica à estrutura `RD_STATS` que o
próprio codificador acabou de calcular —, a guarda desligada é byte-idêntica à
linha de base e o H9c habilitado sem limiar definido não perturba o H9a.

O contraste entre este desempenho e o que o codificador viria a medir é o assunto
das seções seguintes, e ele é acentuado. Em simulação de oráculo o H9c aparece
como a solução mais promissora da tese; em codificação real, isolado do H9a, o
mesmo modelo entrega **5,6% de redução de tempo** na média de quatro sequências
CTC, e apenas **0,6%** na sequência TimeLapse. Este descolamento entre a promessa
do oráculo e o ganho do codificador é um padrão já documentado neste projeto e
constitui, por si, um resultado metodológico: a simulação de oráculo mede a
fração de nós elimináveis e o custo teórico associado, mas não captura a
redistribuição de trabalho que a busca real efetua quando um candidato desaparece.

> **Procedência.** `docs/SINTESE_resultados_metodologia.md` §5 (critério de
> validação por oráculo e critério de integração em C); `docs/ANDAMENTO_tese.md` §7
> (61,2% a 0,20% de perda de SPLIT; paridade dos atributos 0–35 e verificação por
> construção dos atributos 36–38); `results/thesis/M3_protocolo_avaliacao.md` §3.7
> (cascata de critérios e cota superior de 1%); `docs/INVENTARIO_solucoes.md` §4.

---

## 3.3 O fator de confusão e a sua correção

Esta subseção relata o episódio central da caracterização do H9c: um resultado
consolidado internamente foi **retratado** depois que se identificou um fator de
confusão nas medições que o sustentavam. O relato é feito por inteiro, com a
quantificação do efeito e a lista explícita das afirmações retiradas, pois a
auditabilidade da cadeia de decisões é o fio condutor metodológico desta tese e um
episódio de autocorreção medido é evidência desta auditabilidade, não exceção a
ela.

**A causa.** A compilação de desempenho do codificador executa o podador H9a
**sempre** em quadros intraquadro. A função `student_prune_partition` é chamada por
`av1_prune_partitions_before_search` sob a guarda puramente geométrica
`try_student_prune`, **sem** qualquer sinalizador de habilitação, ao contrário do
H9c, cuja ativação depende de `AV1_STUDENT_H9C_ENABLE`. Os primeiros roteiros de
teste do H9c definiram apenas as variáveis de ambiente do H9c e deixaram o H9a nos
seus **limiares compilados por padrão**, `tau_none = tau_split = 0,9`. Então, cada
linha rotulada como H9c media, na verdade, **H9a a 0,9 empilhado com o H9c**, e
não o H9c isolado. A assimetria de projeto entre as duas soluções — uma com guarda
de habilitação e a outra sem — é a origem exata do fator de confusão.

**A quantificação.** A correção consistiu em repetir os mesmos pontos de operação
com o H9a explicitamente neutralizado, através dos limiares τ = 2/2/−1, sendo que
o valor 2 nunca dispara, pois as probabilidades da função softmax são menores ou
iguais a 1, e o terceiro limiar já era inerte. A única variável alterada entre as
duas execuções é esta neutralização. Sobre a sequência Neon1224, com `cpu-used=0`
e contra a mesma âncora, obteve-se o quadro seguinte.

| Configuração | Taxa BD | Redução de tempo |
|---|--:|--:|
| H9c a τ=0,95, medido antes (H9a a 0,9 + H9c) | 0,267% | 17,15% |
| H9c a τ=0,95, isolado | 0,037% | **2,96%** |
| H9c a τ=0,90, medido antes (H9a a 0,9 + H9c) | 0,270% | 17,36% |
| H9c a τ=0,90, isolado | 0,037% | **4,23%** |
| H9c a τ=0,60, medido antes (H9a a 0,9 + H9c) | 0,386% | 20,53% |
| H9c a τ=0,60, isolado | 0,100% | **9,31%** |

O H9c isolado **poda muito pouco**, entregando de 2,96% a 9,31% de redução de
tempo conforme o limiar, e **de 82% a 96% do tempo antes atribuído a ele provinha,
na verdade, do H9a** rodando por baixo. Além disso, o mesmo H9a sozinho, medido
nesta sequência com o baseline que faltava, entrega 0,312% de taxa BD a 17,10% de
redução de tempo, ou seja, praticamente todo o ganho da configuração empilhada.

**A generalização.** A quantificação acima repousava sobre uma única sequência, e
esta era exatamente a fragilidade que a campanha E4 existia para cobrir. O ponto
de operação a τ=0,90 foi repetido, com o H9a neutralizado, em três sequências CTC
além da Neon1224. A fração do tempo indevidamente atribuída ao H9c é de **75%** na
Neon1224, **58%** na PierSeaSide, **28%** na Tango e **95%** na TimeLapse, com
média de **64%**; a configuração empilhada entrega, na média das quatro
sequências, 14,6% de redução de tempo, e o H9c isolado entrega **5,6%**, uma
diferença de 9,0 pontos percentuais. O padrão, deste modo, replica-se, e a
conclusão robusta é a direção e a ordem de grandeza do efeito, não o valor central
exato, pois a dispersão de 28% a 95% mostra que **não se trata de um viés
constante que se pudesse descontar por um fator único**: a contribuição real do
H9c varia de desprezível a moderada conforme o conteúdo. Cabe registrar que os
deltas de tempo aqui envolvidos, de 5 a 13 pontos percentuais, estão muito acima
do piso de ruído posteriormente medido por repetição, cuja resolução é de cerca de
0,46 ponto percentual, e a conclusão não depende desta incerteza.

**As afirmações retiradas.** Por conta do fator de confusão, ficam retiradas e não
reaparecem em nenhum ponto desta tese: (i) a afirmação de que **o H9c seria de duas
a quatro vezes mais eficiente que o H9a**, que era leitura direta das tabelas
pré-busca contaminadas; (ii) a afirmação de que **o H9c superaria o podador nativo
em eficiência**, de mesma origem; e (iii) o veredito original de que **o H9c não
sobreviveria ao piloto real**, com a conclusão associada de que nem mesmo o
contexto de taxa-distorção mais informativo se traduziria em vantagem de tempo de
parede. Este terceiro item merece registro particular, pois é uma retratação em
sentido **favorável** à solução: o veredito apoiava-se inteiramente num piloto de
dois quadros da sequência Jockey, no qual o H9c a τ=0,95 aparecia com 0,264% de
taxa BD a 19,43% de redução de tempo contra 0,194% a 21,58% do H9a, comparação em
que o H9a dominava nos dois eixos. Mas as duas linhas daquela tabela mediam
**objetos diferentes**, uma vez que a linha do H9c continha o H9a por baixo, e a
comparação não era válida. A decisão de parada tomada naquele momento era correta
**com a informação então disponível**, e é sob esta qualificação que ela consta do
registro histórico do projeto.

Uma última observação de ordem técnica cabe aqui. Nas duas sequências em que o H9c
isolado quase não poda, a taxa BD medida fica **levemente negativa** — −0,021% na
PierSeaSide e −0,024% na TimeLapse. São valores minúsculos, mas exatos, pois
contagem de bytes e PSNR são determinísticos, e a explicação é estrutural: podar
um candidato altera o contexto de vizinhança das decisões seguintes, e a busca de
partição do AV1 não é monotônica neste sentido.

> **Procedência.** `docs/ANDAMENTO_tese.md` §7 (retratação no topo) e §8.1
> (identificação e quantificação do fator de confusão na Neon1224; lista das
> afirmações retiradas); `docs/RESULTADOS_BLOCO7_E1_E4.md` §2, §2.1 e §2.2 (campanha
> E4, generalização a quatro sequências, média de 64% e dispersão);
> `docs/RESULTADOS_BLOCO7_E3_DEC_E2.md` §3 (piso de ruído medido, resolução de
> ~0,46 pp); `docs/SINTESE_resultados_metodologia.md` §5;
> `docs/INVENTARIO_solucoes.md` §4.2;
> `src/aom/av1/encoder/partition_strategy.c:2306–2311` (`try_student_prune`, guarda
> geométrica sem sinalizador) e `:2137`. Artefatos: linhas `h9ciso_*` em
> `results/benchmark/fase6/raw_results.csv` (não versionado). Scripts:
> `src/scripts/fase6/{encode_h9c_iso.py, encode_h9adef.py, report_bloco7.py}`.
> *Lacuna:* `[completar: a decomposição do fator de confusão cobre quatro das oito
> sequências CTC; a extensão às quatro restantes não foi executada]`.

---

## 3.4 A substituição direta limpa nas oito sequências CTC

O resultado positivo do H9c não está na composição com o H9a, e sim no cenário de
**substituição direta**: o H9c como **único** podador intraquadro, no lugar da rede
convolucional nativa do codificador, a cada nível de *preset*. Este experimento
neutralizou o H9a desde o próprio roteiro, através dos mesmos limiares τ = 2/2/−1,
e desligou a rede convolucional nativa, de modo que a única diferença entre os
braços comparados, dentro de um mesmo nível de `cpu-used`, é o podador de
particionamento; todas as demais características de velocidade são partilhadas, e
é isto que torna as diferenças intra-nível atribuíveis ao podador. A campanha
soma 192 codificações, distribuídas em seis configurações, oito sequências e
quatro níveis de quantização, e a tabela seguinte apresenta o agregado sobre a
grade CTC completa, contra a âncora `cpu-used=0`.

| *Preset* | Podador | Taxa BD (%) | Redução de tempo (%) | Aceleração |
|:--:|---|--:|--:|--:|
| 1 | rede convolucional nativa | 0,449 | 32,59 | 1,508× |
| 1 | H9c a τ=0,90 | 0,448 | 31,65 | 1,498× |
| 1 | H9c a τ=0,95 | **0,414** | 30,34 | 1,469× |
| 2 | rede convolucional nativa | 0,536 | 42,72 | 1,788× |
| 2 | H9c a τ=0,90 | 0,539 | 42,07 | 1,787× |
| 2 | H9c a τ=0,95 | **0,516** | 40,73 | 1,746× |
| 3 | rede convolucional nativa | **2,722** | 67,94 | 3,159× |
| 3 | H9c a τ=0,90 | 3,397 | 70,67 | 3,474× |
| 3 | H9c a τ=0,95 | 3,384 | 70,25 | 3,419× |

Na média da grade completa, o H9c e a rede convolucional nativa **empatam** nos
*presets* 1 e 2, e o H9c é claramente pior no *preset* 3. Os testes pareados por
sequência sustentam exatamente esta leitura, e não uma leitura mais forte. Com
n = 8 e teste t bilateral, a diferença de taxa BD do H9c a τ=0,95 contra a nativa
no *preset* 1 é de −0,034 ponto percentual na grade completa, com p = 0,278 e
vantagem em 4 de 8 sequências, e no *preset* 2 é de −0,020 ponto percentual, com
p = 0,291 e vantagem em 5 de 8 — ou seja, **nada é significativo na grade
completa, e a afirmação defensável é de paridade, não de superioridade**. No
*preset* 3 a substituição é significativamente pior, com +0,662 ponto percentual,
p = 0,004 e nenhuma das oito sequências favorável. Para referência, todas as
substituições análogas com o H9a são significativamente **piores** que a nativa em
todos os regimes, com diferenças de +0,47 a +1,63 ponto percentual, p ≤ 0,01 e
nenhuma sequência favorável, o que situa o H9c como o único podador desta tese
competitivo neste cenário.

Mas a média sobre a grade, tomada como resumo único, anula um efeito real e
de sinal oposto entre as duas alavancas, e este efeito é uma contribuição própria.
Decompondo por regime de quantização, no regime de alta qualidade — níveis 20 e 32
— o H9c a τ=0,95 custa **menos da metade** da taxa BD da nativa nos dois *presets*
testados: 0,065% contra 0,153% no *preset* 1 e 0,173% contra 0,259% no *preset* 2.
A vantagem é estatisticamente significativa e **replica em dois níveis
independentes** de *preset*: −0,088 ponto percentual com p = 0,043 e vantagem em 6
de 8 sequências no *preset* 1, e −0,086 ponto percentual com p = 0,015 e vantagem
em 7 de 8 sequências no *preset* 2. No regime de baixa qualidade a vantagem
desaparece, com diferenças de +0,009 e +0,027 ponto percentual, ambas não
significativas.

O mecanismo desta dependência é estrutural e foi verificado no código, não apenas
inferido do padrão empírico. Em codificação *All-Intra*, os três podadores nativos
que agem **após** o candidato `PARTITION_NONE` — `av1_ml_predict_breakout`,
`av1_ml_early_term_after_split` e `av1_ml_prune_rect_partition` — estão desligados
por guarda de tipo de quadro, todos sob a condição `!frame_is_intra_only(cm)`.
Então, o nicho que o H9c ocupa, o de decidir depois de conhecer o custo de
taxa-distorção real do `PARTITION_NONE`, é **nativamente vazio** neste regime, o
que explica por que o H9c alcança o empate enquanto o H9a compete de frente com a
rede convolucional intraquadro, que neste regime está ativa. Cabe registrar,
ainda, que a vantagem de qualidade em alta taxa vem acompanhada de custo de tempo:
nos níveis 20 e 32 do *preset* 1 o H9c economiza 20,97% contra 25,26% da nativa,
ou seja, trocam-se cerca de 4,3 pontos percentuais de tempo por cerca de 0,09
ponto percentual de taxa BD, e a fronteira de compromisso é o instrumento correto
para julgar este negócio, não a comparação isolada de taxa BD.

Por fim, a diferença entre os limiares τ = 0,90 e τ = 0,95 é **ruído**, com efeito
médio de −0,01 a −0,03 ponto percentual e violações de sinal em várias sequências,
e os dois não devem figurar como pontos distintos de fronteira. Sobre a fronteira
de Pareto recalculada sobre dezessete configurações completas nas oito sequências,
a configuração `h9c_tau95` no *preset* 1 é não dominada e ocupa taxa BD **abaixo**
do primeiro degrau nativo, 0,414% contra 0,449%, com 93% da redução de tempo dele.

> **Procedência.** `docs/RESULTADOS_fase6_swap_h9c.md` §2 (protocolo, âncora e
> validade da comparação), §3 (agregado sobre a grade CTC), §4 e §4.1
> (decomposição por regime de quantização), §4.2 (mecanismo estrutural e guardas de
> tipo de quadro), §5 (testes pareados), §6 (fronteira de Pareto sobre 8
> sequências), §8 (τ=0,90 contra τ=0,95 é ruído) e §9 (limitações);
> `docs/INVENTARIO_solucoes.md` §4.3; `docs/ANDAMENTO_tese.md` §8.2 e §8.4;
> `src/aom/av1/encoder/partition_search.c:4276`, `:4338` e `:4351` (guardas
> `!frame_is_intra_only`). Artefatos:
> `results/benchmark/fase6_swap_h9c/{raw_results,swap_per_seq,swap_average}.csv` e
> `results/benchmark/fase6_analysis/{cq_decomposition,ts_per_cq,paired_tests,pareto_frontier}.csv`
> (não versionados). Scripts: `src/scripts/fase6/{encode_swap_h9c.py, report_swap.py}`
> e `src/scripts/benchmark/analyze_frontier.py`. *Nota de definição:* os valores de
> redução de tempo desta seção adotam a definição canônica declarada na Seção 3.4
> do Capítulo de Metodologia; a definição alternativa em circulação nos documentos
> do projeto difere de +2,63 a −2,97 pontos percentuais conforme a configuração e
> **reordena** o ranqueamento, razão pela qual as duas não coexistem neste texto
> (`docs/RESULTADOS_fase6_swap_h9c.md` §7).

---

## 3.5 A decomposição de três pernas e a interação medida

A quantificação da Seção 3.3 estabelece que parte do ganho atribuído ao H9c
pertencia ao H9a, mas não estabelece que a atribuição seja uma partição limpa entre
os dois podadores. A decomposição de três pernas resolve esta questão, medindo o
terceiro termo que faltava — o H9a nos seus limiares compilados por padrão, sozinho
e com o H9c desligado — de modo que o balanço fecha na forma
`tempo(H9a) + tempo(H9c) + interação = tempo(H9a + H9c)`. A tabela seguinte
apresenta as reduções de tempo, em quatro sequências CTC.

| Sequência | H9a só | H9c só | H9a + H9c | Soma | **Interação** |
|---|--:|--:|--:|--:|--:|
| Neon1224 | 16,8% | 4,2% | 17,1% | 21,0% | **−3,9 pp** |
| PierSeaSide | 10,4% | 5,4% | 12,8% | 15,8% | **−3,0 pp** |
| Tango | 7,2% | 12,3% | 17,1% | 19,4% | **−2,4 pp** |
| TimeLapse | 9,2% | 0,6% | 11,5% | 9,8% | +1,7 pp |
| **Média** | **10,9%** | **5,6%** | **14,6%** | **16,5%** | **−1,9 pp** |

A interação é **negativa** em três das quatro sequências e negativa na média, com
magnitude de **−1,9 ponto percentual**: os dois podadores empilhados entregam
14,6% de redução de tempo contra 16,5% da soma das partes, ou seja, cerca de
**12% do ganho potencial evapora na sobreposição**. A sequência Tango é o caso
instrutivo, pois é a única em que o H9c sozinho supera o H9a sozinho, 12,3% contra
7,2%, e mesmo ali a interação é negativa. A TimeLapse é a exceção de sinal, com
+1,7 ponto percentual de sinergia, mas o H9c sozinho entrega ali apenas 0,6% de
redução de tempo, valor pequeno demais para sustentar interpretação.

A leitura desta decomposição é a que organiza toda a apresentação dos resultados
de composição desta tese: **alavancas que disputam a mesma ação não se somam**. O
H9a e o H9c ocupam pontos diferentes da dimensão *quando* do espaço de projeto,
pois um decide antes de qualquer avaliação e o outro decide com o custo de
taxa-distorção real em mãos, mas ocupam a **mesma** posição na dimensão *o quê*,
uma vez que ambos perguntam se o nó pode ser encerrado, e por isso caçam os mesmos
blocos de conteúdo liso. A não-aditividade medida aqui, portanto, **não é um limite
informacional**, e sim sobreposição de ação — distinção que a Seção 4 confirma por
contraste, com uma solução que partilha informação idêntica à do H9c e ainda assim
se soma.

> **Procedência.** `docs/RESULTADOS_BLOCO7_E3_DEC_E2.md` §2 (decomposição de três
> pernas, tabela por sequência e interação) e §4 (limitações);
> `docs/ANDAMENTO_tese.md` §8.3 (decomposição aditiva na Neon1224 e correção do
> enunciado da terceira conclusão, de 2026-07-26);
> `docs/SINTESE_resultados_metodologia.md` §6, Conclusão 3 e nota de correção;
> `results/thesis/M1_objeto_e_formulacao.md` §1.4 (as duas dimensões do espaço de
> projeto). Artefato: `results/benchmark/fase6/raw_results.csv` (não versionado).
> Scripts: `src/scripts/fase6/{encode_h9adef.py, report_e3_dec_e2.py}`. *Lacuna:*
> `[completar: a decomposição de três pernas cobre quatro das oito sequências CTC]`.

---

## 3.6 A verificação combinada e a exploração do espaço de limiares

Duas verificações complementares fecham a caracterização do H9c, e ambas
confirmaram as previsões registradas antes da medição. A primeira testou a única
combinação ainda não explorada entre as duas soluções: o H9a em configuração
**conservadora**, com limiares de 0,98 e 0,95 e sem o desligamento das formas
retangulares, empilhado com o H9c, atuando como substituto da rede convolucional
nativa nos *presets* 1, 2 e 3. O resultado, medido sobre a sequência Tango, é que
a combinação **não altera a fronteira de compromisso**: o ponto combinado situa-se
entre a substituição pelo H9c e a substituição pelo H9a, com eficiência
decrescente medida pela razão entre redução de tempo e taxa BD no *preset* 1 —
81,9 para a nativa, 77,2 para o H9c, 67,9 e 65,2 para as duas variantes
combinadas e 29,9 para o H9a balanceado. A rede convolucional nativa permanece no
topo, e a combinação não a domina, o que confirma a leitura da Seção 3.5.

A segunda verificação explorou a faixa de limiares ainda não coberta pela curva de
τ, motivada pela suspeita de que existisse um ponto de operação atrativo entre os
limiares 0,30 e 0,60. Sobre o subconjunto casado de três sequências em que a curva
completa existe, o preço do degrau — pontos percentuais de taxa BD pagos por ponto
percentual de tempo economizado — permanece entre 0,013 e 0,042 na região de
τ = 0,95 a τ = 0,70 e salta para **0,107** a partir de τ = 0,60, de duas e meia a
oito vezes mais caro, o que situa a inflexão da curva na fronteira τ ≈ 0,60–0,70.
O ponto τ = 0,45, medido nas oito sequências, entrega +0,643% de taxa BD a 21,4%
de redução de tempo e é **estritamente dominado** pelo primeiro degrau nativo, que
entrega +0,449% a 32,6%, ou seja, menos taxa BD **e** mais tempo economizado. A
faixa inexplorada foi explorada e não contém nada: é um resultado negativo limpo,
que encerra a questão da curva de limiares.

> **Procedência.** `docs/ANDAMENTO_tese.md` §8.4 (verificação combinada, previsão
> registrada e resultado); `docs/RESULTADOS_BLOCO7_E3_DEC_E2.md` §1 e §1.1 (curva de
> τ, preço do degrau, inflexão e dominância de τ=0,45 pelo degrau nativo);
> `docs/INVENTARIO_solucoes.md` §4.1. Artefatos:
> `results/benchmark/fase6_swap_combo/` e `results/benchmark/fase6/raw_results.csv`
> (não versionados). Scripts:
> `src/scripts/fase6/{encode_swap_combo.py, encode_h9c_cq20.py, report_e3_dec_e2.py}`.

---

## 3.7 Veredito

O veredito sobre o H9c é duplo, e as duas metades precisam ser lidas juntas.

**O H9c não é contribuição autônoma.** A economia de tempo que se lhe atribuía era,
em média, 64% do H9a rodando por baixo; isolado, o podador entrega 5,6% de redução
de tempo na média de quatro sequências, com dispersão de 0,6% a 12,3%; e a
interação com o H9a é negativa em −1,9 ponto percentual, pois as duas soluções
disputam a mesma ação. Como camada extra sobre o H9a no *preset* 0, o H9c não
altera a fotografia, e a configuração empilhada, embora seja um ponto de operação
real e implantável — **+0,171% de taxa BD a 13,6% de redução de tempo** a τ=0,90, e
**+0,160% a 12,6%** a τ=0,95, ambos medidos sobre as oito sequências CTC —, não
pode ser creditada ao podador pós-NONE.

**O H9c é substituto competitivo da rede convolucional nativa nos *presets*
práticos.** Sob substituição direta, com paridade de taxa BD e de tempo nos
*presets* 1 e 2 sobre as oito sequências, com vantagem de qualidade
estatisticamente significativa no regime de alta taxa em dois níveis independentes
de *preset*, e com custo computacional muito menor, pois a inferência do
perceptrone de múltiplas camadas é cerca de cinquenta vezes mais barata por
chamada do que a da rede convolucional nativa, medida em cerca de 486 nanossegundos
contra cerca de 24.700 nanossegundos. A perda ocorre apenas no *preset* 3. Um
perceptrone de trinta e nove atributos alcançar paridade com uma rede convolucional
sintonizada e embarcada no codificador de referência **já é resultado**, e é a
afirmação defensável — superioridade na média da grade não é.

**O H9c é dono do extremo de baixa taxa BD que a escada nativa não alcança.** Os
pontos de menor taxa BD de toda a família medida são as configurações do H9c no
*preset* 0, com +0,160% a +0,171% de taxa BD, menos de um terço do custo do ponto
H9a implantado, que é de +0,568%, e nenhuma das oito sequências excede +0,27%; a
sequência Crosswalk entrega +0,018% de taxa BD a 18,2% de redução de tempo e a
NocturneDance entrega +0,006%, valores que são, na prática, aceleração gratuita. A
escada de *presets* nativos salta do *preset* 0, com 0% de redução de tempo, para
o *preset* 1, com 32,6%, e deixa toda a faixa intermediária descoberta; o valor
prático demonstrado é, deste modo, a **granularidade fina em baixo regime de
aceleração**, e não a superação do pico de eficiência da solução nativa.

O H9c e o H9a, portanto, esgotam a mesma ação, e a decomposição da Seção 3.5
indica, de forma prescritiva, onde procurar ganho adicional: em ações ainda não
disputadas, e não em mais informação de entrada. A solução seguinte foi construída
sobre exatamente esta indicação. Ela partilha o mesmo ponto de inserção pós-NONE e
informação equivalente à do H9c, mas age sobre um conjunto de candidatos disjunto —
as partições estendidas, que consomem 34,3% do tempo de busca local e que nenhuma
das duas soluções anteriores visava — e, por isso, **soma-se** ao H9a em vez de
competir com ele. Os resultados desta solução são apresentados na próxima seção.

> **Procedência.** Consolidação das notas das Seções 3.1 a 3.6, acrescida de:
> `docs/RESULTADOS_BLOCO7_E1_E4.md` §1 e §3 (extremo de baixa taxa BD sobre 8/8
> sequências e a leitura correta da atribuição); `docs/SINTESE_resultados_metodologia.md`
> §6, Conclusões 1, 2 e 3 e o argumento transversal de custo de inferência;
> `docs/RESULTADOS_microbench_pruner.md` (486 ns contra 24.700 ns por chamada);
> `results/thesis/M1_objeto_e_formulacao.md` §1.3 (34,3% do tempo de busca local nas
> partições estendidas) e §1.5 (escada de *presets* nativos). Scripts:
> `src/scripts/fase6/{encode_h9c_cq20.py, report_bloco7.py}`.
