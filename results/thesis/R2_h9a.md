# 2 H9a — poda pré-busca por contexto de taxa-distorção

Esta seção apresenta os resultados da solução **H9a**, a primeira das duas
propostas que sobreviveram a toda a cascata de critérios de decisão e foram
implantadas em C no codificador de referência. São apresentados, nesta ordem, os
critérios de decisão resolvidos fora do codificador, o resultado no conjunto de
teste reservado, a validação universal nas oito sequências das condições comuns
de teste, a substituição direta do podador nativo, a ablação de atribuição
executada dentro do codificador a tempo casado e, por fim, o conjunto de
medições de blindagem que fecha as fragilidades restantes. O H9a é a
concretização da hipótese formulada na Seção 1.2: a decisão de particionamento é
uma decisão de taxa-distorção, e a informação capaz de elevar o desempenho de um
podador aprendido é a informação que a própria decisão utiliza. Todos os
resultados desta seção devem ser lidos contra a âncora declarada no escopo, o
libaom v3.10.0 com `cpu-used=0`, e a definição de redução de tempo em uso é
declarada em cada tabela, uma vez que as duas definições descritas na Seção 3.5
da Metodologia divergem em até cerca de três pontos percentuais.

---

## 2.1 O sinal fora do codificador — critérios 2 e 3

O critério 2 exigia que o conjunto de atributos proposto superasse a heurística
trivial de variância na simulação de oráculo, a risco casado e por margem clara,
antes que qualquer linha de código fosse escrita no codificador. A medição foi
realizada sobre as dez sequências de treino, com sessenta mil superblocos, na
alavanca de comprometimento com `PARTITION_NONE`, que é a alavanca relevante para
o tempo, uma vez que um acerto elimina a subárvore inteira. O risco foi casado
pela perda de decisões de divisão, e a métrica de benefício é a redução de custo
de busca estimada pelo oráculo.

A Tabela 2.1 apresenta a redução de custo por subconjunto de atributos, em três
níveis de risco casado.

**Tabela 2.1** — Redução de custo de busca (%) na simulação de oráculo, por
subconjunto de atributos, a perda de decisões de divisão de 0,5%, 1% e 2%. Dez
sequências de treino, sessenta mil superblocos. Métrica de oráculo; não há
redução de tempo de parede nesta tabela.

| subconjunto de atributos | 0,5% | 1% | 2% |
|---|---:|---:|---:|
| variância | 0 | 0 | 0 |
| `pixels24` (bloco A, vinte e quatro descritores de luminância) | 10,1 | 15,3 | 18,9 |
| **H9a (A + vizinhança + quantização/posição, trinta e seis atributos)** | **15,7** | **20,1** | **24,9** |
| H9c (cota superior, custo RD real do `PARTITION_NONE`) | 33,0 | 33,0 | 39,7 |

A seleção dos valores desta tabela segue uma regra de risco casado: para cada
subconjunto de atributos, toma-se, na varredura de limiares, a linha cujo
`split_lost` fica imediatamente abaixo de cada patamar de risco declarado, e o
`cost_red` correspondente é o valor reportado. Assim, o ponto de 1% de risco do
H9a corresponde a um `split_lost` medido de 0,498%, e o ponto de 2% do
`pixels24` corresponde a 1,434%.

A leitura desta tabela exige uma precisão de composição que foi corrigida por
auditoria posterior e é mantida em todo o texto. O subconjunto `pixels24` **é o
próprio bloco A do H9a**, ou seja, vinte e quatro dos trinta e seis atributos do
H9a são descritores de luminância, e a linha do H9a mede o efeito dos **doze
atributos adicionais** de vizinhança, quantização e posição **sobre** os
descritores de luminância, e não pixels contra outra coisa. Deste modo, o ganho
relativo de aproximadamente 50% do H9a sobre o `pixels24` é o valor de um
acréscimo de contexto de taxa-distorção, cujo custo de extração é desprezível,
pois os dados já estão residentes na memória do codificador no momento da
decisão. A variância entrega zero em todos os três níveis de risco, ou seja, no
regime de risco baixo ela simplesmente não opera.

O critério 3 repetiu a medição no conjunto de validação, formado pelas sequências
HoneyBee, FlowerPan e Lips, com o mesmo casamento de risco. A perda de decisões
de divisão de até 1%, a variância entrega redução de custo próxima de zero e tem
22,6% como menor perda alcançável, os vinte e quatro descritores de luminância
entregam de 26% a 35% com risco mínimo de aproximadamente 0,3%, e o H9a entrega
de **55% a 58%** com risco mínimo de **0,19%**, ou seja, cerca do dobro dos
descritores de luminância. A métrica de segurança confirma a ordenação: o recall
de decisões de divisão do H9a é de 0,98 e 0,82 nos blocos de 64 e de 32 amostras,
contra 0,73 e 0,52 dos descritores de luminância.

Uma ressalva acompanha estes dois critérios e é carregada até o fim do texto. A
simulação de oráculo **superestima o ganho real de tempo**, por um fator próximo
de cinco: uma redução de custo estimada em 35% pelo oráculo corresponde a
aproximadamente 7% de tempo de parede no codificador. Então, o que estas duas
tabelas estabelecem é a **margem relativa** entre subconjuntos de atributos, e
não a magnitude do ganho implantado; o árbitro da magnitude é o critério 5, isto
é, a codificação real no conjunto de teste reservado, apresentada na Seção 2.3.
Além disso, ficou estabelecido em outra família de soluções que a simulação de
oráculo pode inclusive **inverter** a ordenação relativa entre podadores no
codificador real, o que reforça a subordinação destes números ao benchmark de
tempo.

O critério 4, de paridade e inércia, foi atingido nos seus três itens — paridade
bit a bit entre a extração de atributos em C e em Python nos trinta e seis
atributos, ausência de efeito verificada por identidade byte a byte do fluxo de
bits com a guarda de compilação desligada, e aprovação nos testes intraquadro do
codificador —, conforme descrito na Seção 5 da Metodologia.

> **Procedência.** Documentos-fonte: `docs/ANDAMENTO_tese.md` §3 (veredito do
> critério 2, tabela por subconjunto e nota de composição de 26/07) e §4
> (critérios 3 e 4); `docs/SINTESE_resultados_metodologia.md` §4 (cadeia de
> critérios do H9a); `docs/INVENTARIO_solucoes.md` §2.2 e §3.
> Artefatos numéricos: `results/models/gate2_final_sweep.csv` (varredura de τ,
> de onde os valores da Tabela 2.1 são selecionados pela regra de risco casado
> descrita acima; `results/models/gate2_final.csv` reúne uma agregação
> distinta, por limiares fixos de perda de SPLIT, e não é a fonte destes
> números), `results/models/student_h9a/oracle_sim_*.csv`,
> `results/models/student_h9a/gate4_evidence.txt`. Documento da correção de
> composição: `docs/RESULTADOS_auditoria_dominio_pixels.md`.

---

## 2.2 Os pontos de operação e a calibração dos limiares

O ponto de operação da solução **é**, literalmente, um limiar sobre a saída
probabilística do modelo: o podador em C compromete o nó com o `PARTITION_NONE`
quando a probabilidade desta classe excede o limiar correspondente, e força a
divisão quadrada quando a probabilidade da classe de divisão excede o seu. Esta
construção torna a interpretação dos limiares como graus de confiança uma
afirmação verificável, e ela foi verificada em vez de suposta. Sobre **1 816 393
nós de decisão** do conjunto de teste reservado, o erro esperado de calibração da
classe predita é de **0,0112**, valor bom por qualquer padrão da literatura, uma
vez que redes profundas não calibradas situam-se tipicamente entre 5% e 15%.

A precisão real nos limiares é ligeiramente superior ao nominal, o que torna a
descrição conservadora. No limiar implantado de 0,90, a precisão é de **95,6%**
para a classe NONE e de **96,5%** para a classe de divisão; no limiar de 0,95,
sobe para 97,2% e 98,4%. O diagrama de confiabilidade permanece próximo da
diagonal, com desvio sistemático suave apenas no meio da escala, onde o modelo é
sub-confiante em cerca de seis pontos percentuais — o modo de erro benigno, pois
não gera podas erradas confiantes. Cabe ressaltar que esta análise é posterior e
**não realimenta nenhuma decisão de projeto**, além de tomar como referência o
rótulo da decisão de taxa-distorção completa a `cpu-used=0`, de modo que não
captura eventual divergência do que seria correto em regimes de codificação mais
rápidos.

> **Procedência.** Documento-fonte: `docs/RESULTADOS_calibracao.md` §2 e §4.
> Modelo avaliado: `results/models/student_h9a/students.pt`, os mesmos pesos
> exportados para `partition_student_weights.h`. Artefatos:
> `results/models/student_h9a/calibration/{calibration_report.md, ece.csv,
> threshold_precision.csv, reliability.csv}`. Script de reprodução:
> `src/scripts/partition_model/calibration.py`.

---

## 2.3 O conjunto de teste reservado — a curva de operação e os três pilares

O critério 5 foi resolvido no conjunto de teste reservado, formado pelas
sequências Jockey, RaceNight e RiverBank, com dez quadros por sequência, quatro
pontos de quantização e execução em uma única linha de execução. A Tabela 2.2
apresenta a curva de taxa BD contra redução de tempo, por sequência e por ponto
de operação. Nesta campanha, a redução de tempo é reportada como
`(1 − 1/aceleração)·100`, calculada ponto a ponto, e não pela definição ponderada
pelo tempo descrita na Seção 3.5 da Metodologia.

**Tabela 2.2** — Curva do H9a no conjunto de teste reservado, política completa
(comprometimento com NONE, divisão quadrada forçada e poda das formas
retangulares). Taxa BD sobre PSNR-Y; redução de tempo como `(1 − 1/aceleração)·100`.

| ponto | Jockey | RaceNight | RiverBank |
|---|---|---|---|
| P0 | 0,19% / 21,6% / 1,28× | 0,27% / 18,1% / 1,22× | 0,003% / 17,4% / 1,21× |
| P_ref | 0,92% / 32,6% / 1,48× | 0,74% / 31,8% / 1,47× | 0,13% / 24,2% / 1,32× |
| A2 | 1,37% / 47,7% / 1,91× | 1,17% / 41,7% / 1,72× | 0,23% / 30,2% / 1,43× |
| A3 | 2,03% / 57,2% / 2,34× | 1,72% / 50,7% / 2,03× | 0,38% / 36,8% / 1,58× |

Dois pontos de operação sintetizam a curva. O **ponto conservador**, na vizinhança
de P_ref, entrega cerca de **0,6% de taxa BD a aproximadamente 30% de redução de
tempo**, ou seja, aceleração de cerca de 1,4×; o **ponto agressivo**, A3, entrega
cerca de **1,4% de taxa BD a aproximadamente 48% de redução de tempo**, ou seja,
aceleração próxima de 2,0×, ambos como médias das três sequências. A configuração
intermediária P_rect, que veio a ser a implantada nas campanhas seguintes,
registra 0,464% de taxa BD a 26,49% de redução de tempo. O comportamento é
consistente nas três sequências, e a RiverBank é o caso extremo favorável, com
0,003% de taxa BD a 17,4% de redução de tempo, ou seja, aceleração praticamente
gratuita.

O segundo pilar é a atribuição do ganho ao aprendizado. Sob política idêntica,
variando apenas a fonte do escore, o modelo domina o escore aleatório em todas as
sequências e em todos os níveis medidos: a aceleração casada de 1,15×, o modelo
custa 0,13% contra 1,79% do aleatório na Jockey, 0,34% contra 3,75% na RaceNight e
0,11% contra 3,40% na RiverBank; a aceleração casada de 1,30×, custa 0,26% contra
3,21% na Jockey e 0,90% contra 6,01% na RaceNight. Deste modo, o ganho não provém
da poda em si, e sim da seleção de quais nós podar.

O terceiro pilar é a comparação contra a heurística de variância, e é nele que
está o veredito honesto desta fase. **A forma estrita do critério 5 não foi
atingida.** Ela exigia dominância em taxa BD a aceleração casada em ao menos duas
das três sequências, e no conjunto de teste reservado as faixas de aceleração dos
dois braços saíram **disjuntas nas três sequências**, de modo que não existia par
casado a comparar: o modelo opera de 1,03× a 1,59× com taxa BD de 0,008% a 1,68%,
enquanto a variância parte de 1,23× a 2,13× e nunca desce de 0,75% a 3,96% de taxa
BD. A causa é conhecida e está registrada: a grade de limiares foi congelada na
calibração feita sobre a HoneyBee, onde a variância a limiar 0,95 rendia 1,34× e
havia sobreposição, e o conteúdo de teste poda de forma mais agressiva sob o mesmo
limiar. A grade **não foi estendida no teste**, pois estender configuração vendo
dado de teste violaria o congelamento antisseleção.

O que sustenta a contribuição, então, são três afirmações, e nenhuma delas depende
da sobreposição de faixas. A primeira é a **redução de tempo contra a âncora**,
forte e consistente nas três sequências, nos termos da Tabela 2.2. A segunda é a
**superioridade sobre o escore aleatório**, clara nas três sequências. A terceira,
e a mais forte, é a **atribuição a política casada**: mantida idêntica a política
primária de comprometimento com NONE e variando somente a fonte do escore, a taxa
BD mínima alcançável pelo escore do modelo é de **0,16%** na Jockey, **0,09%** na
RaceNight e **0,008%** na RiverBank, contra **1,76%**, **3,96%** e **0,75%** da
variância, razões de **11×, 44× e 94×**. Ou seja, o escore do modelo alcança uma
região de baixa taxa BD que o escore de variância é estruturalmente incapaz de
tocar, pois a sua regra grosseira compromete o nó com o `PARTITION_NONE` em
qualquer bloco liso, inclusive naqueles que deveriam ser divididos. Na validação,
onde a grade sobrepôs, a dominância direta a aceleração casada foi obtida — 0,66%
contra 3,53% a 1,5× e 1,41% contra 4,92% a 1,75×.

> **Procedência.** Documentos-fonte: `docs/RESULTADOS_fase5.md` §1 a §5;
> `docs/ANDAMENTO_tese.md` §4.1 (Fase 5 completa e veredito do critério 5);
> `docs/INVENTARIO_solucoes.md` §3.1 (médias das três sequências).
> Artefatos numéricos: `results/benchmark/h9_test/{Jockey,RaceNight,RiverBank}/`
> `{curve_safe,curve_aggr,ablation}/{summary.csv,curve.csv}` (não versionados).
> Scripts de reprodução: `results/benchmark/{fase5_final,matched_bd}.py` e
> `src/scripts/benchmark/analyze_ablation.py`.

---

## 2.4 A validação universal nas oito sequências CTC

O conjunto de teste reservado pertence ao mesmo universo de dados do aprendizado
de máquina, e por isso a validação externa foi executada sobre as **oito
sequências da Classe A1 das condições comuns de teste** (do inglês *Common Test
Conditions* – CTC) da AOM, em 4K, 10 bits e quinze quadros, com a mesma grade de
quantização. A Tabela 2.3 apresenta os dois pontos de operação do H9a contra a
âncora e contra os *presets* nativos. **Todas as reduções de tempo desta tabela
usam a definição canônica**, padronizada na Seção 3.5 da Metodologia; sob a
definição ponderada pelo tempo, os mesmos artefatos registram, por exemplo,
30,42% para o `cpu-used=1` em vez de 32,59%, e 34,14% para o ponto agressivo em
vez de 31,51%.

**Tabela 2.3** — Resultados finais na CTC, média sobre as oito sequências da
Classe A1. Âncora libaom `cpu-used=0`; taxa BD sobre PSNR-Y; redução de tempo na
definição canônica.

| configuração | taxa BD | redução de tempo | aceleração |
|---|--:|--:|--:|
| H9a equilibrado (P_rect) | +0,568% | 17,72% | 1,223× |
| H9a agressivo (A3) | +1,403% | 31,51% | 1,492× |
| libaom `cpu-used=1` | +0,449% | 32,59% | 1,508× |
| libaom `cpu-used=2` | +0,536% | 42,72% | 1,788× |
| libaom `cpu-used=3` | +2,722% | 67,94% | 3,159× |

A pergunta prática que esta tabela responde é a do profissional que já dispõe do
botão de velocidade do codificador, e a resposta é desfavorável ao ponto de
aprendizado de máquina. O *preset* `cpu-used=1` entrega mais economia de tempo que
o H9a equilibrado — 32,59% contra 17,72% — a taxa BD **menor**, e também domina o
H9a agressivo, oferecendo praticamente a mesma redução de tempo a menos de um
terço da taxa BD. Em sete das oito sequências o *preset* nativo fica à frente.
Cabe registrar que a rede convolucional nativa de poda intraquadro está ligada nos
*presets* `cpu-used` 1, 2 e 3 e desligada na âncora, o que foi verificado no código
de *speed features* e por teste empírico de identidade byte a byte, de modo que a
comparação é entre configurações limpas e disjuntas. **Nenhum ponto de aprendizado
de máquina proposto domina a rede convolucional nativa na fronteira de taxa BD
contra tempo.**

O valor prático medido é outro, e é mensurável. A escada dos *presets* é discreta
e esparsa: a transição de `cpu-used=0` para `cpu-used=1` já salta para
aproximadamente 33% de redução de tempo, e não existe degrau intermediário. O H9a,
por ser controlado por uma grade contínua de limiares, cobre continuamente a faixa
de aproximadamente **12% a 22% de redução de tempo** que o botão nativo não
oferece, e nela o custo é baixo — na NocturneDance, o ponto equilibrado registra
**+0,15% de taxa BD a 12,6% de redução de tempo**, abaixo do que o nativo
interpolado ofereceria naquele ponto. Deste modo, a contribuição não precisa
vencer a fronteira nativa inteira para ser útil: ela preenche uma granularidade de
operação em baixo regime de aceleração que o degrau grosseiro dos *presets* não
alcança. Cabe destacar, por fim, que este achado atinge a utilidade prática e não
a validação metodológica, uma vez que a superioridade do escore de taxa-distorção
sobre a variância sob política casada, estabelecida na Seção 2.3, permanece
inalterada.

> **Procedência.** Documentos-fonte: `docs/RESULTADOS_fase6.md` §1 a §3;
> `docs/ANDAMENTO_tese.md` §4.2 (desenho pré-registrado da Fase 6);
> `docs/INVENTARIO_solucoes.md` §1 e §3.2. Artefatos numéricos:
> `results/benchmark/fase6/{raw_results.csv, bdrate_per_seq.csv,
> bdrate_average.csv, tables.tex}` e
> `results/benchmark/fase6_analysis/ts_definitions.csv`. Scripts de reprodução:
> `src/scripts/fase6/{run_fase6.sh, encode_ctc.py, report_ctc.py}` e
> `src/scripts/fase6/analyze_frontier.py`.

---

## 2.5 A substituição direta do podador nativo

A comparação da Seção 2.4 responde à pergunta do profissional, mas não isola o
mérito do podador, pois os *presets* nativos combinam a rede convolucional de poda
com dezenas de heurísticas que nada têm de aprendizado de máquina. A comparação de
categoria correta mantém tudo constante e troca **apenas** o podador: a
`cpu-used` fixo, a rede convolucional nativa é desligada por uma variável de
ambiente dedicada e o H9a assume como único podador intraquadro. O mecanismo foi
verificado por identidade de resumo criptográfico do fluxo de bits em quatro
configurações, confirmando que a ausência da variável é inerte e que a troca de
fato substitui um podador pelo outro.

A Tabela 2.4 apresenta o resultado, na definição canônica de redução de tempo.

**Tabela 2.4** — Substituição direta do podador nativo pelo H9a, a `cpu-used`
fixo. Âncora libaom `cpu-used=0`; oito sequências da Classe A1; redução de tempo
na definição canônica.

| `cpu-used` | podador | taxa BD | redução de tempo | aceleração |
|:--:|---|--:|--:|--:|
| 1 | rede convolucional nativa | +0,449% | 32,59% | 1,508× |
| 1 | H9a equilibrado | +0,915% | 40,20% | 1,694× |
| 1 | H9a agressivo | +1,685% | 51,82% | 2,104× |
| 2 | rede convolucional nativa | +0,536% | 42,72% | 1,788× |
| 2 | H9a equilibrado | +1,030% | 50,05% | 2,046× |
| 2 | H9a agressivo | +1,805% | 60,97% | 2,610× |
| 3 | rede convolucional nativa | +2,722% | 67,94% | 3,159× |
| 3 | H9a equilibrado | +3,866% | 73,09% | 3,754× |
| 3 | H9a agressivo | +4,347% | 77,30% | 4,465× |

O padrão é consistente nos três níveis: no mesmo `cpu-used`, a rede convolucional
nativa isolada tem sempre taxa BD menor **e** redução de tempo menor que o H9a, ou
seja, não há dominância direta em nenhum nível, e sim um compromisso genuíno em
que o H9a corta mais tempo ao custo de taxa BD desproporcional, da ordem de duas a
três vezes a da nativa. A análise dos nove pontos por dominância mostra que apenas
**um deles é estritamente dominado** — o H9a equilibrado a `cpu-used=1`, cravado
pelo *preset* nativo `cpu-used=2`, que tem simultaneamente taxa BD menor e redução
de tempo maior. Mas a pertinência formal à fronteira não deve ser lida como
qualidade operacional: o retorno marginal, medido como pontos percentuais de
redução de tempo por ponto percentual de taxa BD entre pontos consecutivos, é de
**116,4** no trecho puramente nativo de `cpu-used=1` para `cpu-used=2`, cai para
**14,8** ao acrescentar o H9a equilibrado e chega a **2,7** no pior trecho. Deste
modo, a fronteira de dominância, tomada isoladamente, **superestima** o quanto o
H9a é competitivo.

A leitura honesta deste resultado é que o H9a **não supera** a rede convolucional
nativa como podador, nem a `cpu-used=0` contra os *presets*, nem isolando o
podador a `cpu-used` fixo. Cabe registrar, ainda, que uma leitura alternativa
chegou a ser proposta e foi **retirada por medição posterior**: a razão de cerca
de cinquenta vezes entre o custo de inferência por chamada dos dois modelos
caracteriza o algoritmo de decisão isolado e não constitui vantagem do podador
implantado, pois o custo próprio do podador dentro do codificador — extração de
atributos e inferência somadas — é de no máximo **0,32% do tempo de codificação**,
com 0,16% a 0,21% para a rede convolucional nativa e 0,26% a 0,32% para o H9a. O
custo de inferência não é alavanca em direção nenhuma, e o que produz o ganho de
tempo são as decisões de poda.

> **Procedência.** Documentos-fonte: `docs/RESULTADOS_fase6.md` §4, §4.1, §4.2 e
> §4.3, incluindo a correção de 2026-07-19 sobre o custo de inferência;
> `docs/RESULTADOS_microbench_pruner.md` §6; `docs/INVENTARIO_solucoes.md` §3.3.
> Artefatos numéricos: `results/benchmark/fase6_swap/{raw_results.csv,
> swap_per_seq.csv, swap_average.csv, swap_tables.tex}`. Scripts de reprodução:
> `src/scripts/fase6/{run_swap.sh, encode_swap.py, report_swap.py}`.

---

## 2.6 A ablação de atribuição dentro do codificador, a tempo casado

A afirmação de atribuição da Seção 2.3 repousava sobre região alcançável, e não
sobre pares casados, precisamente porque as faixas de aceleração saíram disjuntas
no conjunto de teste reservado. O experimento **E5** foi desenhado para fechar
essa lacuna no lugar metodologicamente legítimo — o conjunto de **validação**,
cujo papel declarado no protocolo é escolher limiares operacionais — e produziu a
**primeira comparação a tempo casado contra a variância em toda a investigação**.
Os três braços rodam a mesma política no mesmo codificador, com comprometimento
com NONE puro, divisão quadrada nunca forçada e poda das formas retangulares
desligada, variando exclusivamente a fonte do escore. Apenas a grade de limiares
da variância foi estendida ao extremo conservador, até 0,999, jamais sondado
antes; a grade do modelo permaneceu **intocada**, de modo que a alteração
beneficia o adversário e não a hipótese. A campanha cobriu trinta e quatro pontos
de operação, cento e quarenta e quatro codificações e cerca de vinte e duas horas.

Na **FlowerPan** as faixas se sobrepõem e o modelo domina em todos os pares
medidos, sem interpolação. A aceleração de aproximadamente 1,15×, o modelo custa
0,095% de taxa BD contra 0,437% da variância, razão de **4,6×**; a aproximadamente
1,27×, custa 0,638% contra 1,180%, razão de **1,85×**. Contra o escore aleatório
as razões são muito maiores, de **158×** a aproximadamente 1,10× e de **19×** a
aproximadamente 1,19×. A grade estendida fez exatamente o que se pretendia: a
variância a limiar 0,999 entrega 1,7% de redução de tempo a **0,000%** de taxa BD,
ou seja, a heurística **consegue** operar em regime conservador, e isto passou a
ser medido em vez de suposto — a comparação ficou mais honesta, e não mais
favorável.

Na **Lips** o resultado é de natureza diferente e mais informativa. Entre os
limiares 0,99 e 0,97 a variância salta de 1,006× para **3,563×** de aceleração, e
de 0,019% para **6,58%** de taxa BD, sem qualquer ponto intermediário na grade; o
modelo vive inteiramente dentro deste vão, de 1,074× a 1,283×, de modo que a
comparação a tempo casado contra a variância **não existe** nesta sequência.
Contra o escore aleatório a comparação existe e é limpa, com razão de **7,0×** a
aproximadamente 1,19×. O mecanismo provável é a natureza do conteúdo — um
primeiro plano de rosto, com grandes regiões de pele lisas e de variância baixa e
homogênea, sobre as quais um limiar de variância se comporta de forma
aproximadamente bimodal —, mas cabe registrar que este mecanismo é interpretação e
não medição. A consequência para a tese é forte: a não-sobreposição observada no
conjunto de teste reservado é **propriedade do escore**, e não artefato da grade
congelada, o que fecha a explicação alternativa incômoda que a Fase 5 deixara em
aberto.

Um resultado adicional merece destaque: o modelo atinge **taxa BD negativa em duas
de duas sequências** — −0,015% a 3,8% de redução de tempo na FlowerPan e −0,073% a
6,9% na Lips —, ou seja, existe um ajuste em que ele economiza de 4% a 7% do tempo
com ganho marginal de qualidade. A variância também chega a 0,000%, mas por não
podar praticamente nada, com 0,2% a 1,7% de redução de tempo contra 3,8% a 6,9% do
modelo.

O veredito é declarado sem contorno. **O critério estrito não foi atingido**: ele
pedia dominância a tempo casado sobre a variância em ao menos **duas de três**
sequências, e obteve-se **uma de duas** — a FlowerPan —, pois na Lips não há par
casado a comparar e a HoneyBee foi cortada por **decisão de escopo declarada**,
por ser a sequência em que a grade original havia sido calibrada e, portanto, a
menos independente das três. A afirmação defensável não é a de dominância
universal, e sim a seguinte: sob política idêntica, o escore do modelo gradua
continuamente a região implantável de baixa taxa BD, que o escore da variância ou
não alcança ou atravessa de um salto, e, onde ambos coexistem, o modelo custa de
duas a cinco vezes menos taxa BD pelo mesmo tempo. Três pontos de limiar na Lips,
dentro do vão, decidiriam o critério na forma estrita a um custo aproximado de
~2,3 h, e permanecem não executados.

> **Procedência.** Documento-fonte: `docs/RESULTADOS_E5_ablacao_validacao.md` §1 a
> §6; `docs/DECISOES_escopo.md` (corte da HoneyBee);
> `docs/INVENTARIO_solucoes.md` §2.1 (atualização de 28/07 e a contradição
> remanescente entre `pixels24` e variância, que o E5 **não** arbitra). Artefatos
> numéricos: `results/benchmark/e5_ablation/{FlowerPan,Lips}/{curve,runs}.csv`
> (não versionados). Scripts de reprodução:
> `src/scripts/benchmark/ablation_attrib.py`, `run_e5_validation.sh` e
> `stop_e5_after_lips.sh`.

---

## 2.7 A blindagem do Bloco 7 — a curva de limiares e a resolução do tempo

O conjunto de campanhas designado Bloco 7 existe para fechar as fragilidades que
uma banca examinadora atacaria, e três dos seus resultados incidem diretamente
sobre a leitura desta seção. O primeiro localiza o **joelho da curva de limiares**
por medição. Sobre o subconjunto casado de sequências em que toda a curva existe,
o preço do degrau — pontos percentuais de taxa BD pagos por ponto percentual de
redução de tempo ao afrouxar o limiar — permanece entre 0,013 e 0,042 na faixa que
vai de 0,95 até 0,70, e salta para **0,107** a partir de 0,60, ou seja, de duas
vezes e meia a oito vezes mais caro. Deste modo, a região plana termina em 0,70, e
o joelho situa-se na fronteira de **0,60 a 0,70**, e não dentro do vão que se
supunha.

O segundo resultado é um negativo limpo. O limiar de 0,45, sondado justamente por
se supor que houvesse um ponto de operação escondido na faixa inexplorada, não está
sobre uma inflexão favorável: a reta que liga os limiares 0,60 e 0,30 prevê 0,395%
de taxa BD na redução de tempo de 18,16% que ele atinge, e o valor medido é de
**0,430%**, ou seja, 0,035 ponto percentual **acima** da interpolação, pois a curva
é levemente convexa naquele trecho. Medido nas oito sequências, este ponto registra
**+0,643% de taxa BD a 21,4% de redução de tempo**, contra **+0,449% a 32,59%** do
*preset* nativo `cpu-used=1` — isto é, **estritamente dominado pelo botão nativo**,
com taxa BD maior e redução de tempo menor. A faixa inexplorada foi explorada e não
contém nada, o que confirma por medição a previsão registrada no plano de que
nenhuma extrapolação de limiar a `cpu-used=0` alcançaria a fronteira nativa. Cabe
ressaltar que esta curva de limiares é a da configuração composta, que empilha o
podador pós-NONE sobre o H9a nos seus padrões compilados, e a decomposição dessa
composição é objeto da próxima seção.

O terceiro resultado é o mais consequente para a leitura de todas as tabelas desta
seção: a **resolução medida da comparação pareada de tempo**. Cinco repetições
intercaladas de três configurações sobre quatro pontos de quantização, em execução
contínua, dão coeficiente de variação mediano de **0,28%** e máximo de 0,64% no
tempo bruto. O número que governa a leitura é o desvio da redução de tempo pareada:
para o ponto equilibrado, cuja redução medida é de 20,29%, o desvio sobre as cinco
repetições é de **±0,23 ponto percentual**, o que estabelece uma resolução efetiva
de aproximadamente **0,46 ponto percentual**, tomada como dois desvios; para o
*preset* nativo `cpu-used=1`, com redução medida de 32,84%, o desvio é de ±0,09
ponto e a resolução, de cerca de 0,18 ponto. O plano original supunha desvio da
ordem de 1% a 2% por codificação, de modo que a estimativa anterior era pessimista
por um fator próximo de **quatro**. A consequência é simétrica e deve ser
explicitada: diferenças de redução de tempo antes descartadas como indistinguíveis
do ruído são, de fato, **resolvíveis**, e o ganho marginal de 1,02 ponto percentual
da segunda solução implantada situa-se a cerca de 4,4 desvios acima do ruído. A
ressalva pertinente é que o desvio medido é intra-execução, obtido numa mesma
janela contínua, na mesma sequência e no mesmo contêiner, de forma que não captura
deriva entre dias ou estados térmicos distintos; como as campanhas desta tese
rodaram em janelas contínuas análogas, é o desvio pertinente para as comparações
internas, mas comparar números medidos com semanas de intervalo exige cautela
adicional.

> **Procedência.** Documentos-fonte: `docs/RESULTADOS_BLOCO7_E3_DEC_E2.md` §1,
> §1.1, §3 e §4; `docs/RESULTADOS_BLOCO7_E1_E4.md` §1 e §4;
> `results/thesis/M3_protocolo_avaliacao.md` §3.6 (mesma medição, na
> Metodologia). Artefatos numéricos: `results/benchmark/fase6/raw_results.csv` e
> `results/benchmark/fase6_repeat/raw_results.csv` (não versionados). Scripts de
> reprodução: `src/scripts/fase6/encode_h9c_cq20.py --taus 45`,
> `src/scripts/fase6/encode_repeat.py --seq Crosswalk --reps 5` e
> `src/scripts/fase6/report_e3_dec_e2.py`.

---

## 2.8 Síntese e encaminhamento

O H9a está, então, caracterizado em todas as réguas do protocolo. O sinal de
taxa-distorção supera a heurística trivial fora do codificador por margem de
aproximadamente 50% sobre os descritores de luminância, o que motivou o custo caro
da integração em C; no conjunto de teste reservado, a solução entrega cerca de
0,6% de taxa BD a aproximadamente 30% de redução de tempo no ponto conservador e
cerca de 1,4% a aproximadamente 48% no ponto agressivo, com o ganho atribuível ao
modelo por duas vias independentes; nas oito sequências das condições comuns de
teste, nenhum ponto de aprendizado de máquina domina a rede convolucional nativa,
e o valor medido reside na granularidade fina de 12% a 22% de redução de tempo que
a escada discreta dos *presets* não oferece. Os vereditos não atingidos foram
declarados como tais, com o que de fato se obteve em cada caso.

Duas questões ficam abertas por construção e organizam as seções seguintes. A
primeira é a de quanto do ganho atribuído às configurações compostas pertence de
fato ao H9a pré-busca, uma vez que a curva de limiares da Seção 2.7 empilha um
segundo podador sobre os padrões compilados do primeiro. A segunda é a de se o
contexto de taxa-distorção **real**, disponível somente depois que o codificador
avaliou o `PARTITION_NONE` e que a Tabela 2.1 registra como cota superior de
aproximadamente o dobro do H9a, converte-se em ganho implantável ou permanece como
promessa da simulação de oráculo. A próxima seção apresenta o H9c, o refinamento
pós-NONE, o fator de confusão que inflou a sua leitura inicial e o resultado limpo
que sobrou depois da decomposição.

> **Procedência.** Consolidação das notas das Seções 2.1 a 2.7; nenhum valor novo
> é introduzido nesta síntese.
