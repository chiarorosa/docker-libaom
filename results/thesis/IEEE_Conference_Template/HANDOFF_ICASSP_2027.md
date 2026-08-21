# HANDOFF — artigo ICASSP 2027 (RPP)

> **Arquivo temporário de retomada.** Registra onde o trabalho parou em 2026-08-20 e o que
> falta para escrever o artigo. Apagar quando o `.tex` estiver submetido.
> Commit de referência: `ec1d919` em `ml-partition-dev` (já empurrado).

---

## 1. O que é este artigo

Segundo artigo da tese, para o **ICASSP 2027**. Não é artigo de solução: o objeto é
**representação e informação**, não aceleração de codificador.

**Chamada de trabalhos.** Até **4 páginas de conteúdo técnico incluindo figuras e
referências**, mais uma **5ª página opcional contendo apenas** referências,
agradecimentos de financiamento e declaração de conformidade ética. A tática é ocupar as 4
páginas com texto e figuras e empurrar as referências inteiras para a 5ª.

**Nome escolhido:** **RPP — Representation Probe for Partitioning**.
**Título escolhido (forma de achado):** *Causal Encoder State Outperforms Deep Pixel
Representations in AV1 Intra Partition Decisions*.

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

## 5. Estrutura acordada do artigo

- **I. Introduction** — M1 a M8 do `CLAUDE.md` B.4. A lacuna (M6): os trabalhos existentes
  propõem uma política e reportam velocidade; nenhum separa a qualidade da fonte de
  pontuação da agressividade da política, nem mede onde reside a informação preditiva.
- **II. The Partition Decision in AV1** — dez formas, ordem fixa de avaliação e, o ponto
  próprio deste artigo, **que informação já foi paga em cada instante**. Fecha com
  delimitação de escopo (intra, `cpu-used=0`, ação NONE-commit isolada).
- **III. RPP: Representations and Measurement Protocol**
  - III-A as representações, declaradas por **acesso à informação**, não por arquitetura.
  - III-B o protocolo: política fixa, varredura de τ, modelo de custo `candidatos × n²`,
    `regret_abs(n) = none_rdcost(n) − RD_subtree(n) ≥ 0` normalizado — equação numerada.
    Declarar que **não é tempo de parede**.
  - III-C dataset e partição, com declaração antivazamento.
  - III-D **a justiça do braço profundo** (princípio 3): corpus correto, sem vazamento,
    capacidade dobrada e os dois objetivos. Aqui entra a inversão do α.
- **IV. Results and Discussion** — tabela da fronteira; escada com as razões; a
  decomposição B contra C; limitações como resultado.
- **V. Conclusions** — fórmula B.10, sem número inédito.

**Decisões de tratamento já tomadas:**
- A inversão do objetivo (α=3 melhor que α=0) **entra** no artigo, em III-D, como prova de
  tratamento justo.
- Os **quatro** degraus entram na tabela, incluindo que **A+B domina A+B+C**.

## 6. O que falta — a fazer amanhã

1. **Escrever o `.tex`** em `results/thesis/IEEE_Conference_Template/`, seguindo o
   `CLAUDE.md` daquele diretório (Parte A normativa, Parte B de estilo). Reservar no texto
   o espaço e o propósito das figuras, sem gerá-las ainda.
2. **Gerar as figuras por último**, no contêiner `av1_bench` com `build/venv-ml`, em inglês
   e paleta acadêmica sóbria. Duas previstas:
   - Fig. 1 — linha do tempo de disponibilidade de informação dentro do nó. **Desenho
     novo**, não reaproveitar a Fig. 2 do LASCAS.
   - Fig. 2 — curvas de perda de otimalidade × redução de custo casada, eixo y logarítmico,
     com barra de amplitude das sementes nos degraus. É a figura que carrega o artigo.
3. **Checklist da Parte C** do `CLAUDE.md`, compilar e conferir a paginação (4 técnicas + 5ª
   só de referências, financiamento e ética).
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
