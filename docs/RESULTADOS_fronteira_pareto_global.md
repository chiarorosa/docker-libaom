# Fronteira de Pareto global recomposta — 8 sequências CTC, com o H9d

**Data:** 2026-07-29. **Branch:** `ml-partition-dev`. **Motivo:** a lacuna registrada
em `ANDAMENTO_tese.md §0.3` e em `SINTESE_resultados_metodologia.md §6` — *"a fronteira
Pareto global está desatualizada — é de 3 seqs e não contém o H9d"*. Este documento
registra a recomposição, o que ela mudou e o que continua em aberto.

---

## 1. O que estava errado no registro

A pendência estava **mal descrita**, e a recomposição custou uma reexecução de script,
não uma campanha de codificação. Dois fatos:

1. **O artefato já era de 8 sequências.** O `results/benchmark/fase6_analysis/pareto_frontier.csv`
   de 19/07 traz `native_cpu1` a **0,4487% / 32,594%**, que é o valor canônico das oito
   sequências CTC, e não o de três sequências (0,368% / 28,6%) que os documentos
   descreviam. A recomposição para oito sequências já havia ocorrido; o texto é que
   continuou citando a análise anterior.
2. **O artefato estava desatualizado, não incompleto.** O `fase6/raw_results.csv` é de
   **27/07** e já continha as quatro configurações do H9d; o `pareto_frontier.csv` era de
   **19/07**, oito dias anterior. O `analyze_frontier.py` descobre as configurações
   sozinho (`:131`) e mantém as completas nas oito sequências, de modo que **bastava
   reexecutá-lo**.

---

## 2. Comando de reprodução

Dentro do contêiner, sem nenhuma codificação nova (~1 min):

```bash
cd /workspace && ./build/venv-ml/bin/python src/scripts/fase6/analyze_frontier.py
```

Entradas (padrão do script): `results/benchmark/fase6/raw_results.csv`,
`fase6_swap/raw_results.csv`, `fase6_swap_h9c/raw_results.csv`. Saídas reescritas em
`results/benchmark/fase6_analysis/`: `pareto_frontier.csv`, `cq_decomposition.csv`,
`paired_tests.csv`, `ts_definitions.csv`, `ts_per_cq.csv`.

Âncora: libaom `cpu-used=0`. Redução de tempo na definição **canônica**. Taxa BD por
PSNR-Y (Bjøntegaard). Oito sequências CTC Classe A1, quinze quadros, grade CQ
20/32/43/55.

---

## 3. A fronteira recomposta

A fronteira passa de **17 para 24 configurações** avaliadas, das quais **15 são
não-dominadas**, do menor ao maior custo em taxa BD:

| configuração | taxa BD | redução de tempo | aceleração |
|---|--:|--:|--:|
| `h9c_tau95` (cpu0) | 0,160% | 12,61% | 1,148× |
| `h9c_tau90` (cpu0) | 0,172% | 13,59% | 1,162× |
| `h9c_tau95_cpu1` | 0,414% | 30,34% | 1,469× |
| `h9c_tau90_cpu1` | 0,448% | 31,65% | 1,498× |
| `native_cpu1` | 0,449% | 32,59% | 1,508× |
| `h9c_tau95_cpu2` | 0,516% | 40,73% | 1,746× |
| `native_cpu2` | 0,536% | 42,72% | 1,788× |
| `h9a_bal_cpu2` | 1,030% | 50,05% | 2,046× |
| `h9a_aggr_cpu1` | 1,685% | 51,82% | 2,104× |
| `h9a_aggr_cpu2` | 1,805% | 60,97% | 2,610× |
| `native_cpu3` | 2,722% | 67,94% | 3,159× |
| `h9c_tau95_cpu3` | 3,384% | 70,25% | 3,419× |
| `h9c_tau90_cpu3` | 3,397% | 70,67% | 3,474× |
| `h9a_bal_cpu3` | 3,866% | 73,09% | 3,754× |
| `h9a_aggr_cpu3` | 4,347% | 77,30% | 4,465× |

---

## 4. Os três achados

### 4.1 O extremo de baixo custo em taxa BD melhorou, e a Conclusão 2 se fortalece

Os dois pontos de menor taxa BD de toda a fronteira são do H9c a `cpu-used=0`, e agora
custam **0,160% e 0,172%** de taxa BD a 12,61% e 13,59% de redução de tempo. A leitura
anterior, de três sequências, registrava 0,213% e 0,227% a 13,9% e 15,0%. O regime
descoberto pela escada nativa — que salta de 0% para 32,59% de redução de tempo entre
`cpu-used=0` e `cpu-used=1` — continua sendo preenchido apenas pelas soluções
aprendidas, e a custo menor do que se supunha.

Registre-se que estes dois pontos **não figuravam na fronteira de 19/07**: naquela
execução eles não estavam completos nas oito sequências e foram excluídos pelo filtro
do script (`:133`). A Conclusão 2, portanto, ganhou os seus donos legítimos.

### 4.2 Os pontos do H9d entram na fronteira, e são dominados

As quatro configurações do H9d passam a figurar na análise, e **nenhuma delas é
não-dominada** no espaço global:

| configuração | taxa BD | redução de tempo | dominada por |
|---|--:|--:|---|
| `ml_balanced` (base, sem H9d) | 0,568% | 17,72% | `native_cpu1`, `native_cpu2`, `h9c_*_cpu1`, `h9c_*_cpu2` |
| `ml_bal_h9d` (implantado) | 0,586% | 18,74% | o **mesmo** conjunto |
| `ml_bal_h9d_pl20` | 0,651% | 19,81% | idem, mais `h9c_tau45` |
| `ml_aggr` (base) | 1,403% | 31,51% | `h9a_bal_cpu1/2`, `h9c_*_cpu2`, `native_cpu1/2` |
| `ml_aggr_h9d` | 1,409% | 31,68% | idem |
| `ml_aggr_h9d_pl20` | 1,420% | 32,16% | idem |

**Esta dominância não contradiz o resultado do H9d, e a leitura correta importa.** O
H9d é um **complemento** do H9a, medido como contribuição **marginal sobre uma base
fixa** e testado quanto à não-dominância **contra a curva de limiares do próprio H9a**
(`RESULTADOS_H9d_CTC.md`; memória de projeto `compare-levers-to-deployed-h9a`). Nesse
quadro, que é o válido, o H9d entrega **+1,02 pp de redução de tempo por +0,018 pp de
taxa BD** e vence a curva de limiares em 6 de 8 sequências.

O que a tabela acima mostra é outra coisa: a **base** já é dominada. O `ml_balanced`
sozinho é dominado exatamente pelo mesmo conjunto de configurações que domina o
`ml_bal_h9d`. Ou seja, o H9d **não perde posição alguma** — ele herda a posição do H9a,
melhorando-a marginalmente dentro dela. A dominância vem de o ponto de operação viver
em `cpu-used=0`, regime no qual os presets nativos a `cpu-used` 1 e 2 entregam mais
redução de tempo por menos taxa BD.

Deste modo, a conclusão da tese não muda, mas passa a ser enunciável de forma mais
precisa: **nenhum ponto do H9a nem do H9a somado ao H9d é não-dominado na fronteira
global**; o valor destas soluções é a granularidade fina dentro da curva de limiares, e
não uma posição na fronteira global.

### 4.3 A Conclusão 1 se mantém, com um dono a mais no extremo inferior

Nenhum ponto de aprendizado de máquina domina a rede convolucional nativa. O
`native_cpu1` e o `native_cpu2` permanecem não-dominados, com o `h9c_tau95_cpu1` colado
ao primeiro (0,414% / 30,34% contra 0,449% / 32,59%) — menos taxa BD e menos redução de
tempo, ou seja, empate técnico sem dominância em direção alguma.

---

## 5. Efeito sobre a lacuna registrada

A lacuna **L1** de `results/thesis/A3_RETRATACOES_E_LACUNAS.md` — *"a fronteira de
compromisso global não contém os pontos do H9d"* — está **fechada para o regime
`cpu-used=0`**, que é onde o H9d foi medido e implantado.

**Continua em aberto:** o H9d empilhado nos níveis `cpu-used` 1, 2 e 3, que não foi
codificado. Fechá-lo custaria duas bases × três níveis × oito sequências × quatro
pontos de quantização, ou seja, cerca de **192 codificações**. Há razão medida para
esperar rendimento baixo: o H9d mostrou-se **inerte sobre a base agressiva** do H9a
(+0,17 pp de redução de tempo, acima da resolução em apenas 1 de 8 sequências), e os
presets nativos mais rápidos já podam agressivamente, de modo que o resíduo sobre o
qual o H9d age tende a encolher.

---

## 6. Limitações

- A fronteira mistura configurações medidas em campanhas distintas (`fase6`,
  `fase6_swap`, `fase6_swap_h9c`), todas contra a mesma âncora `cpu-used=0` e a mesma
  grade de quantização, mas **não na mesma janela contínua de execução**. A resolução do
  tempo pareado de ~0,46 pp (`RESULTADOS_BLOCO7_E3_DEC_E2.md §4`) vale dentro de uma
  janela; diferenças menores que isso entre campanhas pedem cautela adicional. Os
  empates técnicos apontados em §4.3 estão nessa faixa.
- As configurações incompletas nas oito sequências são excluídas pelo script e não
  figuram na fronteira; entre elas, `h9ciso_*` e `h9adef`, que são diagnósticos de
  isolamento e não pontos de operação.
- A definição de redução de tempo é a **canônica**. O `ts_definitions.csv` da mesma
  execução registra a divergência para a definição ponderada pelo tempo, que chega a
  −2,97 pp no `h9c_tau95_cpu2` e +2,63 pp no `ml_aggr`.

---

## 7. Procedência

Artefatos reescritos em `results/benchmark/fase6_analysis/` (execução de 2026-07-29):
`pareto_frontier.csv` (24 configurações, 15 não-dominadas), `ts_definitions.csv`,
`cq_decomposition.csv`, `paired_tests.csv`, `ts_per_cq.csv`. Entradas:
`results/benchmark/{fase6,fase6_swap,fase6_swap_h9c}/raw_results.csv`. Script:
`src/scripts/fase6/analyze_frontier.py`. Documentos afetados por esta recomposição:
`ANDAMENTO_tese.md §0.3` e §8.3, `SINTESE_resultados_metodologia.md §6`,
`results/thesis/R6_analise_integrada.md` e `A3_RETRATACOES_E_LACUNAS.md` (lacuna L1).
