# Campanha D2 enxuta — H9d empilhado sobre o H9a substituto da CNN nativa (cpu-used 1–3)

Data: 2026-10-04. Objeto: medir o marginal do H9d (poda seletiva das partições estendidas) quando o
H9a balanceado substitui a rede convolucional nativa de poda de partição nos *presets* 1, 2 e 3, e
ler o resultado contra as previsões **pré-registradas** em
`results/thesis/IEEE_Conference_Template/ISCAS/CLAUDE.md` §6 (commit `28f9145`, 2026-10-03 14:12,
anterior à primeira codificação, lançada às 14:14). Fecha a lacuna declarada em
`results/thesis/R6_analise_integrada.md` §6.1 (H9d medido só em `cpu-used=0`).

## 1. Desenho

- Grade CTC A1: 8 sequências 4K 10 bits, 15 quadros, `cq-level` 20/32/43/55, comando de
  `encode_ctc.encode` (inalterado; `--tile-columns=1 --threads=2 --row-mt=0`).
- Dois braços por (sequência, cq, preset), executados em sequência no **mesmo** binário
  `build/libaom_perf_h9d` (Release, `-DPARTITION_ML_STUDENT=1`), CNN nativa desligada
  (`AV1_DISABLE_NATIVE_CNN=1`):
  `h9a_bal_cpuN` (base remedida, H9d desligado) e `h9a_bal_h9d_cpuN` (base + H9d nos limiares
  compilados PL10: 0,0910 / 0,1031 / 0,0144). 192 codificações.
- Base agressiva excluída por decisão registrada antes da medição (marginal de +0,17 pp em p0,
  `R4_h9d.md` §4.7).
- Âncora: linhas `anchor` (`cpu-used=0`) de `results/benchmark/fase6/raw_results.csv`.
- Execução: 2026-10-03 14:14 → 2026-10-04 01:46 (-0300), contêiner `av1_bench`, 0 erros.

## 2. Integridade (P0) — cumprida

As 96 linhas-base reproduzem **byte a byte** (bytes e PSNR-Y) as linhas `h9a_bal_cpuN` de
`results/benchmark/fase6_swap/raw_results.csv` (julho/2026): 96/96 idênticas. Consequência: o H9d
desligado é inerte no binário usado, e toda diferença entre os braços é atribuível ao H9d. Os BD-BR
da base coincidem com os publicados (0,915 / 1,030 / 3,866%); o TS da base difere de julho em
+0,07% em média por codificação (faixa −5,01% a +1,86%), o que justifica o pareamento na mesma
campanha.

## 3. Resultados (média de 8 sequências; TS canônica; BD-BR PSNR-Y)

| preset | base BD-BR | base TS | +H9d BD-BR | +H9d TS | ΔBD (pp) | ΔTS (pp) | t pareado ΔTS | seq. ΔTS > 0 |
|:--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| 1 | 0,915% | 40,28% | 0,947% | 41,15% | +0,032 (p = 0,080) | **+0,87** | p = 0,001 | 8/8 |
| 2 | 1,030% | 50,16% | 1,069% | 50,99% | +0,039 (p = 0,026) | **+0,83** | p = 0,001 | 8/8 |
| 3 | 3,866% | 73,10% | 3,868% | 72,96% | +0,002 (p = 0,820) | **−0,14** | p = 0,007 | 0/8 |

Por sequência: `results/benchmark/fase6_swap_h9d/marginal_per_seq.csv`.

## 4. Leitura contra o pré-registro

| id | previsão | obtido | veredito |
|---|---|---|---|
| P0 | 96 linhas-base byte-idênticas | 96/96 | cumprida |
| P1 | 0,46 < ΔTS < 1,02 pp em p1 e p2 | +0,87 e +0,83 pp | cumprida |
| P2 | ΔBD ≤ +0,05 pp em p1, p2 e p3 | +0,032, +0,039, +0,002 pp | cumprida |
| P3 | ΔTS(p3) < min(ΔTS p1, p2) | −0,14 < 0,83 | cumprida |

Nenhuma recalibração foi feita após a leitura.

## 5. Interpretação

1. **O H9d soma sobre o substituto da CNN em p1 e p2**: +0,87 e +0,83 pp de TS, positivo nas 8
   sequências, a +0,032 e +0,039 pp de BD-BR. Eficiência marginal ΔTS/ΔBD de 27 e 21 pp/pp. Para
   referência, afrouxar os limiares do H9a de balanceado para agressivo na mesma substituição custa
   15,1 pp/pp em p1 e 14,1 em p2 (pontos de `fase6_swap`, julho): o H9d é cerca de 1,8 e 1,5 vez
   mais eficiente. Ressalva: os dois termos da razão vêm de campanhas distintas.
2. **Em p3 o ganho desaparece** (−0,14 pp, 0/8 positivas), exatamente onde o codificador passa a
   podar as partições estendidas por conta própria (`prune_ext_part_using_split_info=1`,
   `speed_features.c`, *speed* ≥ 3): mesma ação, sobreposição — confirmação da regra de composição
   por um caminho independente do par H9a+H9c. O sinal negativo é consistente entre sequências,
   mas a magnitude fica abaixo da resolução de 0,46 pp; atribuí-lo ao custo da inferência do H9d
   sem poda correspondente é hipótese não medida.
3. **Fronteira** (17 configurações sem os pontos de p0 do LASCAS e sem τ_stop = 0,90): 13 pontos
   não dominados, **10 do NPL-AV1** e 3 nativos. O ponto H9a balanceado + H9d em p2
   (1,069% / 50,99%) é não dominado; em p1 é dominado pelo nativo p2 e em p3 pela base.

## 6. Limitações

1. O ΔTS é pareado e intra-campanha; a comparação de eficiência com o botão de limiar (§5.1)
   mistura esta campanha com `fase6_swap` (julho).
2. Tempo de parede, não o *user time* exigido pela CTC v9 §5.7 (as colunas `user_s`/`sys_s` foram
   gravadas, mas a âncora não tem *user time*).
3. Apenas a base balanceada; o H9d sobre o pós-NONE (H9c) não foi medido.
4. Os artefatos de `results/benchmark/` não são versionados.

## 7. Reprodução (contêiner `av1_bench`)

```bash
build/venv-ml/bin/python src/scripts/fase6/encode_swap_h9d.py                  # 192 codificações
build/venv-ml/bin/python src/scripts/fase6/encode_swap_h9d.py --check-integrity  # P0
build/venv-ml/bin/python src/scripts/fase6/report_swap_h9d.py                  # §3–§4
```
