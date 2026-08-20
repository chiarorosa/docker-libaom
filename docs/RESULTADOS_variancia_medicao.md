# Variância de medição do tempo de codificação

Este documento quantifica a dispersão entre repetições da medição de tempo de
parede e, sobretudo, a dispersão da grandeza derivada que é efetivamente
reportada — a economia de tempo (do inglês *time savings*, TS) da Eq. (1) do
artigo. O número existe para responder a uma pergunta de revisão que o texto
até então não respondia: quantas repetições foram feitas e qual é a resolução
de medição contra a qual as diferenças reportadas devem ser lidas.

## 1. Procedência e protocolo

O artefato é `results/benchmark/fase6_repeat/raw_results.csv`, produzido pela
mesma infraestrutura da campanha principal (`results/benchmark/fase6/`). Ele
cobre **cinco repetições** (`_r1` a `_r5`) de três configurações — a âncora, o
podador de primeiro estágio nos limiares balanceados e o `cpu-used=1` nativo —
sobre **uma sequência** (Crosswalk), nos quatro pontos de quantização 20, 32,
43 e 55, com quinze quadros, como na campanha principal.

Comando de reprodução, no contêiner `av1_bench`:

```
python3 src/scripts/benchmark/analyze_repeat_variance.py \
    results/benchmark/fase6_repeat/raw_results.csv --anchor anchor
```

## 2. Dispersão do tempo de parede

Coeficiente de variação sobre as cinco repetições, por configuração e ponto de
quantização:

| configuração | CQ 20 | CQ 32 | CQ 43 | CQ 55 |
|---|---|---|---|---|
| âncora | 0,253% | 0,422% | 0,247% | 0,403% |
| 1.º estágio, balanceado | 0,316% | 0,527% | 0,217% | 0,348% |
| `cpu-used=1` | 0,189% | 0,128% | 0,191% | 0,637% |

O máximo observado é 0,637%, ou seja, **toda medição de tempo absoluto varia
menos de 0,7%** entre execuções.

## 3. Dispersão do TS derivado

O TS é calculado pela Eq. (1), isto é, a razão por ponto de qualidade antes de
qualquer média. Duas formas de pareamento foram percorridas: a pareada, em que
a repetição *i* da configuração é comparada com a repetição *i* da âncora; e a
cruzada, que percorre as vinte e cinco combinações e, por não supor pareamento
algum, dá o limite pessimista.

| configuração | modo | média | desvio-padrão | mínimo | máximo |
|---|---|---|---|---|---|
| 1.º estágio, balanceado | pareado | 20,294% | **0,230 pp** | 20,010% | 20,528% |
| 1.º estágio, balanceado | cruzado | 20,294% | 0,247 pp | 19,952% | 20,898% |
| `cpu-used=1` | pareado | 32,843% | 0,092 pp | 32,707% | 32,964% |
| `cpu-used=1` | cruzado | 32,843% | 0,159 pp | 32,549% | 33,148% |

## 4. Leituras

**A resolução de medição é de 0,23 ponto percentual de TS.** É este o número que
o artigo passa a citar, e ele qualifica duas afirmações que antes eram apenas
asseridas. A contribuição marginal do segundo estágio sobre a base balanceada,
de 1,02 ponto percentual, está **4,4 vezes acima** desta resolução e, portanto,
sobrevive à dispersão. Já a contribuição marginal sobre a base agressiva, que o
artigo descreve como abaixo da resolução de medição, passa a ter a expressão
quantificada em vez de retórica.

**A razão é mais estável do que os termos que a compõem.** A campanha de
repetição correu cerca de 2% mais lenta em termos absolutos do que a campanha
principal — a âncora em CQ 20 mede 666,39 s aqui contra 652,56 s na fase6 —, e
ainda assim o TS pareado difere em apenas 0,11 pp do valor de execução única
registrado em `fase6/bdrate_per_seq.csv` para a mesma configuração (20,176%),
ou seja, meio desvio-padrão. Isto é evidência direta a favor do desenho da
Eq. (1): a razão por ponto de qualidade cancela a deriva de carga da máquina,
que é comum aos dois braços.

**O pareamento importa pouco.** A diferença entre o desvio pareado e o cruzado
é de 0,017 pp para o primeiro estágio e 0,067 pp para o `cpu-used=1`. A
conclusão não depende, deste modo, da hipótese de que as repetições partilham a
deriva.

## 5. Limitações registradas

- O estudo cobre **uma única sequência** (Crosswalk). A dispersão aqui medida é
  de execução, e não captura a dispersão entre conteúdos, que é reportada
  separadamente pela Tabela I do artigo e é uma ordem de grandeza maior.
- A configuração repetida é o **primeiro estágio isolado** nos limiares
  balanceados, e não o ponto balanceado implantado, que empilha o segundo
  estágio. Não há razão para esperar que o segundo estágio altere a dispersão de
  execução, uma vez que o seu custo de inferência é inferior a um terço de um
  por cento do tempo, mas isto não foi medido.
- As repetições correram no mesmo hospedeiro e na mesma imagem de contêiner. A
  dispersão entre máquinas ou entre versões de imagem não está coberta.
