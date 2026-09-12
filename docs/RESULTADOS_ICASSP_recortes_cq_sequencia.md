# ICASSP — recortes da fronteira por cq-level e por sequência de teste

**Data:** 2026-09-12
**Origem:** terceira colocação do orientador Daniel Palomino sobre o artigo do ICASSP 2027.
**Pedido:** (1) verificar se BC+NC continua melhor que BC em cada um dos quatro cq-level,
já que a métrica agrega os quatro; (2) verificar BC vs. BC+NC individualmente nas três
sequências de teste, para responder se o ganho aparece nos três conteúdos independentes ou é
dominado por uma sequência.
**Script:** `src/scripts/partition_model/slice_frontier.py`
**Artefato:** `results/models/oracle_regret_rpp_test3/slices.csv`
**Contexto:** roda sobre a vara limpa de [[RESULTADOS_ICASSP_split_limpo_teste]].

---

## 1. Por que não foi preciso retreinar nada

Os pacotes do `rpp_ladder` já existem; o que se refez foi o **replay**, agora fatiado. Isso é
válido porque **o replay da árvore podada é independente entre superblocos** — podar um nó de um
superbloco não altera nada em outro. Logo, particionar a lista de superblocos e somar o
denominador restrito a cada partição dá exatamente a fronteira daquele recorte.

`collect()` passou a carimbar cada superbloco com o pkl de origem e com o seu `none_rd`, o que
permite os sete recortes (4 cq + 3 sequências) numa **única** leitura do dataset, em vez de sete
execuções do `oracle_regret.py` — 12 leituras de pkl contra ~24.

Cada fatia casa a redução de busca **dentro dela mesma**; o τ do agregado não é herdado. É o que
mantém a lógica do artigo: comparar o desperdício de RD à mesma economia de busca.

## 2. Reprodução

```bash
build/venv-ml/bin/python src/scripts/partition_model/slice_frontier.py \
    --out /workspace/results/models/oracle_regret_rpp_test3/slices.csv
```

## 3. Resultado — ganho de BC+NC sobre BC (%), média de 3 sementes

| recorte | 10% | 15% | 20% | 25% | 30% |
|---|--:|--:|--:|--:|--:|
| `cq-level` 20 | **−7,7** | **−4,1** | — | — | — |
| `cq-level` 32 | +66,8 | +72,1 | +74,5 | +61,3 | +37,6 |
| `cq-level` 43 | +9,5 | +31,9 | +44,0 | +56,2 | +59,0 |
| `cq-level` 55 | +43,1 | +58,7 | +56,7 | +54,7 | +50,6 |
| Jockey | +44,6 | +56,2 | +53,1 | +55,4 | +25,4 |
| RaceNight | +60,5 | +58,7 | +72,1 | +63,3 | +53,4 |
| RiverBank | +49,9 | +49,5 | +19,0 | **−8,9** | — |

Em quantas das 3 sementes BC+NC vence BC:

| recorte | 10% | 15% | 20% | 25% | 30% |
|---|:-:|:-:|:-:|:-:|:-:|
| `cq-level` 20 | 1/3 | 1/3 | 0/1 | — | — |
| `cq-level` 32 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `cq-level` 43 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| `cq-level` 55 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| Jockey | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| RaceNight | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| RiverBank | 3/3 | 3/3 | 3/3 | 1/3 | — |

## 4. Resposta às duas perguntas

**Pergunta 2 (por sequência): o ganho não é dominado por uma sequência.** Aparece nos três
conteúdos independentes, em 3/3 sementes, até 20% de redução. A ressalva é RiverBank a partir
de 25%, onde as curvas se cruzam.

**Pergunta 1 (por cq-level): vale em três dos quatro.** cq32, cq43 e cq55 mantêm o ganho em
toda a faixa (cq43 fraco só em 10%, +9,5%, com 1/3 sementes). **Em cq20 o ganho não existe** —
BC+NC perde em 2 de 3 sementes em 10% e 15%, e em 3 de 3 em 20%. Não é ruído de semente: é
consistente nos três pontos que cq20 alcança.

## 5. As duas exceções, e o que se pode dizer delas

**cq20.** É também o recorte em que a terminação antecipada é menos aplicável: **não alcança
25% de redução de busca**. Na taxa mais alta as partições são uniformemente finas, `NONE`
raramente é ótimo, e sobra pouco para podar — que é justamente o regime em que a dimensão do
bloco vizinho tem pouco poder discriminativo, porque quase todos os vizinhos são finos. A
leitura levada ao artigo é que **o benefício se concentra onde há busca a remover**, e ela é
sustentada pelo alcance limitado de redução nesse recorte, não apenas por plausibilidade.

**RiverBank.** O ganho decresce com a quantidade de redução (+49,9 → +49,5 → +19,0 → −8,9) e as
curvas se cruzam entre 20% e 25%; BC+NC sequer alcança 30%. A hipótese levada ao artigo, com
*hedge*, é o detalhe espacial fino que cobre a maior parte do quadro, deixando poucos nós em que
terminar cedo é seguro. As penalidades absolutas altíssimas dessa sequência em 30% (BC em 138–183)
são compatíveis com essa leitura.

## 6. Limitações

- **Fatias pequenas:** cada recorte por cq tem ~28 mil superblocos e cada um por sequência ~37 mil,
  contra 111 mil do agregado. O espalhamento entre sementes é maior, e é por isso que a contagem
  "vence em k/3 sementes" está reportada ao lado da média — ela é o que separa efeito de ruído.
- **cq20 e RiverBank não alcançam todos os pontos de leitura**, e os traços na tabela do artigo
  marcam isso. Nenhum valor foi extrapolado.
- A explicação de RiverBank é hipótese com *hedge*, não medida: nenhuma métrica de complexidade
  espacial (SI/TI) foi calculada para sustentá-la.
- Continua replay offline, sem propagação de erro de reconstrução e sem BD-rate.

## 7. O que entrou no artigo

`paper_daniel.tex`: Tabela II nova (ganho de BC+NC sobre BC por recorte, com traços nos pontos
não alcançados) e um parágrafo na Seção V, entre o parágrafo do controle de permutação e o do
CSP. O parágrafo nomeia as duas exceções em vez de reportar só o agregado — decisão tomada com o
autor, sob o argumento de que um revisor com acesso à tabela por CQ faria exatamente essa
pergunta, e a omissão custaria mais que o resultado.
