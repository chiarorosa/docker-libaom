# ICASSP — controle de permutação de NC (o passo BC+NC é informação ou dimensionalidade?)

**Data:** 2026-09-10
**Origem:** colocação metodológica do orientador Daniel Palomino sobre o artigo do ICASSP 2027.
**Pergunta:** o ganho de BC+NC sobre BC vem da informação contextual dos vizinhos causais,
ou apenas de a entrada passar de 24 para 32 colunas?
**Scripts:** `src/scripts/partition_model/rpp_ladder.py` (treino) ·
`src/scripts/partition_model/oracle_regret.py` (pontuação) ·
`src/scripts/partition_model/features.py` (`RPP_SUBSETS`, `RPP_SHUFFLE_COLS`, `shuffle_columns`)
**Artefatos:** `results/models/rpp_ladder/A_Bshuf_s{0,1,2}/students.pt` ·
`results/models/oracle_regret_rpp/frontier.csv` (linhas `RPP_A_Bshuf_s*`) ·
`results/models/oracle_regret_rpp_shuf/` (rodada isolada, com report próprio)

---

## 1. O confundidor que o controle remove

O artigo acusa a literatura de definir conjuntamente atributos, modelo, regra de decisão e
quantidade de poda, o que impede atribuir um ganho à informação de entrada. A metodologia do
artigo fixa a regra de terminação e casa a redução de busca — mas o passo BC (24 colunas) para
BC+NC (32 colunas) **também muda a largura da entrada**. Esse é um confundidor dentro da própria
casa, exatamente da classe criticada no parágrafo de lacuna da Seção I.

Até aqui a defesa era indireta: BC+CSP acrescenta 4 colunas e **piora** a penalidade (53 para 64),
logo somar colunas não basta. O argumento é observacional — CSP é outro grupo de variáveis, e o
seu fracasso pode significar "CSP é ruidoso", não "dimensionalidade não ajuda".

## 2. Método — rung `A_Bshuf`

Nova rung da escada de informação, treinada e pontuada pelo mesmo caminho canônico das demais.

- **Colunas:** as mesmas 32 de `A_B` (A = 0..23, B = 24..31 do vetor H9a de 36).
- **Intervenção:** os valores das colunas 24..31 são **reatribuídos entre amostras**, dentro de
  cada nível de bloco, sob **uma única permutação aplicada ao bloco inteiro** das oito colunas.
  Em bloco, e não coluna a coluna: permutar cada coluna isoladamente destruiria também a
  correlação interna do grupo (largura contra altura do vizinho de cima, por exemplo), removendo
  mais do que o confundidor de dimensionalidade que se quer isolar.
- **Onde se aplica:** treino **e** pontuação, com permutações independentes. Um modelo ajustado
  contra um bloco descorrelacionado precisa ser pontuado contra um bloco descorrelacionado;
  apresentar o bloco íntegro a pesos ajustados contra ruído mediria desvio de distribuição, e não
  valor de informação.
- **Mantido idêntico a BC+NC:** largura da entrada (32), arquitetura (32-64-32-3), número de
  parâmetros, receita de treino (30 épocas, AdamW, lr 1e-3, lote 4096), 3 sementes, mesmo corpus
  de treino e mesma vara held-out.
- **Preservado pela permutação:** a distribuição marginal das oito colunas e a sua joint interna.
  **Destruída:** apenas a associação com o nó que elas descrevem.

Sementes de permutação derivadas de (semente, nível): `991000 + seed*1000 + dim` no treino,
`773000 + seed*1000 + dim` na pontuação.

## 3. Reprodução

```bash
# treino das três sementes do controle (contêiner av1_bench)
build/venv-ml/bin/python src/scripts/partition_model/rpp_ladder.py \
    --rungs A_Bshuf --seeds 0 1 2 --out-dir /workspace/results/models/rpp_ladder

# pontuação na vara held-out (coleta enxuta: só braços que leem nd["feat"])
build/venv-ml/bin/python src/scripts/partition_model/oracle_regret.py \
    --arms RPP_A_s0 RPP_A_s1 RPP_A_s2 \
           RPP_A_B_s0 RPP_A_B_s1 RPP_A_B_s2 \
           RPP_A_Bshuf_s0 RPP_A_Bshuf_s1 RPP_A_Bshuf_s2 \
    --out-dir /workspace/results/models/oracle_regret_rpp_shuf
```

Vara held-out: HoneyBee, FlowerPan, Lips, Jockey, RaceNight, RiverBank — 226.447 superblocos e
3.808.703 nós de decisão, os mesmos números declarados no artigo.

## 4. Resultado

Penalidade de custo RD normalizada, em 10^-4 %, média de 3 sementes, a redução de busca casada:

| rung | rótulo no artigo | colunas | 10% | 15% | 20% | 25% | 30% |
|---|---|--:|--:|--:|--:|--:|--:|
| A | BC | 24 | 5,2 | 13,0 | 27,9 | 53,5 | 91,1 |
| **A_Bshuf** | **BC+shuffled NC** | **32** | **5,2** | **12,7** | **27,0** | **48,0** | **85,8** |
| A_B | BC+NC | 32 | 2,3 | 5,2 | 14,9 | 33,4 | 69,4 |

Faixa entre sementes em 25%: BC 49,1–57,2 · BC+shuffled-NC 40,8–53,2 · BC+NC 30,0–36,4.
A faixa do controle **sobrepõe** a de BC e é **disjunta** da de BC+NC. O mesmo vale em 10, 15 e 20%.

Teste de permutação exata, 3 sementes por braço, em 25% (20 partições, piso bilateral p = 0,10):

| confronto | diferença | p |
|---|--:|--:|
| BC+shuffled-NC vs BC+NC | +14,57 | 0,100 (piso) |
| BC vs BC+NC | +20,04 | 0,100 (piso) |
| BC vs BC+shuffled-NC | +5,46 | 0,400 |

Os dois confrontos com BC+NC atingem o menor p alcançável pelo desenho — separação completa dos
grupos. BC contra o controle não separa.

Acurácia de treino (procedência apenas), nível 64x64, média de 3 sementes: BC 0,7801 ·
BC+shuffled-NC 0,7797 · BC+NC 0,8150.

## 5. Conclusão

O controle **passa**: com largura de entrada e capacidade idênticas às de BC+NC, o embaralhamento
devolve o desempenho ao de BC. O ganho de BC+NC é atribuível à associação das oito colunas com o
nó que elas descrevem, e não às dimensões adicionais que ocupam.

Efeito no artigo: a afirmação central deixa de depender de inferência por eliminação e passa a se
apoiar numa intervenção sobre as mesmas colunas, com tudo o mais fixo. A palavra "Controlled" do
título passa a ser sustentada por um controle, e não apenas pelo casamento de Δ.

## 6. Limitações

- **O controle sai marginalmente melhor que BC** (48,0 contra 53,5). As faixas entre sementes se
  sobrepõem, e a diferença é compatível com ruído ou com regularização por injeção de ruído. Isso
  torna o controle **conservador**: concede à hipótese rival o benefício de um número melhor, e a
  hipótese ainda assim falha. Dos 20,1 pontos do passo BC para BC+NC, no máximo 5,5 (27%)
  poderiam ser atribuídos à entrada mais larga, e esses 5,5 estão dentro do espalhamento entre
  sementes. Essa decomposição **não** foi levada ao artigo, por dar aparência de precisão a ruído.
- **Em 30% a separação encolhe:** BC+shuffled-NC 68,6–94,5 contra BC+NC 67,7–71,9, faixas que se
  tocam na borda. O texto do artigo ancora a não-sobreposição em 25%.
- **Poder estatístico baixo:** 3 sementes por braço impõem piso de p em 0,10. O p não desce por
  limitação do desenho, não por fraqueza do efeito.
- **Não decompõe qual parte de NC informa** (disponibilidade, dimensões do vizinho, granularidade,
  anisotropia). Seria outra ablação.
- **Não toca a limitação central do artigo:** continua replay offline sobre custos gravados, sem
  propagação de erro de reconstrução e sem BD-rate.

## 7. Notas de procedência

- As seis rungs antigas (`RPP_A_s*`, `RPP_A_B_s*`) foram **remedidas do zero** nesta rodada e
  reproduziram o `frontier.csv` canônico **byte a byte** (11 pontos de tau cada). É o teste de
  regressão que dá crédito à linha nova.
- `results/models/oracle_regret_rpp/frontier.csv` ganhou 33 linhas (`RPP_A_Bshuf_s*`); o restante
  do arquivo saiu idêntico, verificado por `diff`.
- `rpp_ladder.py` reescreve `training.csv` a cada execução. As linhas das 4 rungs originais foram
  recuperadas do commit `ec1d919` e remescladas; o arquivo hoje cobre as 5 rungs.
- `oracle_regret.py` ganhou `--arms` com coleta enxuta automática: quando a seleção só lê
  `nd["feat"]`, a coleta deixa de materializar `feat_h9a_b1`, `feat_h9c` e a luma do superbloco.
  Sem isso a vara held-out completa projetava ~23,5 GiB contra o limite de 23,41 GiB do contêiner.
  A poda **não altera número algum** — `total_none_rd`, o regret por nó e a política de poda são
  os mesmos.
- A trava de coerência de `plot_loss_frontier_icassp_fig2.py` passou a conferir 50 células (era
  45) e inclui a linha nova. A curva do controle fica **fora do desenho**, pelo mesmo motivo do
  controle aleatório: ela cai sobre a de BC — que é o resultado — e duas curvas sobrepostas numa
  coluna de 86 mm tornam ambas ilegíveis.

## 8. Onde entrou no artigo

`results/thesis/IEEE_Conference_Template/ICASSP/paper_daniel.tex` (cópia de revisão):
Seção III-B ganha o parágrafo do controle; Tabela I ganha a linha `BC+shuffled NC (32 features)`;
Seção V ganha o parágrafo de leitura do controle; o fecho do parágrafo do CSP deixa de ser a prova
contra dimensionalidade e passa a corroboração. O `paper.tex` original **não** foi tocado.

## 9. Robustez a deixar-uma-semente-de-fora

Levantada a hipótese de que a semente 1 do controle seria fora da curva (40,77 em 25%, contra
53,21 e 49,99 das outras duas), a verificação mostra que **ela é a melhor semente nos três
braços**: BC 49,10 contra 57,22 e 54,05; controle 40,77 contra 53,21 e 49,99; BC+NC 30,02 contra
33,86 e 36,38. Não é anomalia do controle, e sim uma inicialização que produz modelos melhores
neste arranjo. Com n=3 tampouco há critério que a classifique como outlier.

Efeito de removê-la, em 25% de redução:

| cenário | BC+NC sobre o controle | controle contra BC |
|---|--:|--:|
| todas as sementes | 30,4% | −10,2% |
| sem a s1 em **todos** os braços | 31,9% | −7,2% |
| sem a s1 **só no controle** (enviesado) | 35,2% | −3,5% |

A remoção consistente praticamente não move nada, e o controle segue abaixo do BC. A remoção
apenas no controle melhora os números, mas retira a melhor semente de um braço mantendo-a nos
outros dois — comparação enviesada e trivialmente reprodutível por quem baixar os artefatos.

Conclusão: a ordenação BC+NC < controle ~ BC sobrevive ao *leave-one-seed-out*, o que é um
resultado de robustez mais forte que a média sobre três sementes isolada.

---

> **Nota de superação (2026-09-12).** Os valores das §4 e §9 deste documento vêm da rodada sobre
> as **seis** sequências held-out. O artigo passou a reportar apenas as **três de teste**, por
> colocação do orientador — ver [[RESULTADOS_ICASSP_split_limpo_teste]]. O método, o desenho do
> controle e as conclusões aqui seguem válidos; sob o conjunto limpo o controle fica **mais**
> forte (BC+NC melhora 41,3% sobre ele em 25%, contra 30,4% aqui).
