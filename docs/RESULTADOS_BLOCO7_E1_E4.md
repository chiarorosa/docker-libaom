# Bloco 7 — E1 (H9c em 8/8 sequências) e E4 (decomposição do confound H9a/H9c)

**Data:** 2026-07-25
**O que são.** Duas campanhas de encode do Bloco 7, rodadas em cadeia serial (28 encodes,
~3h10). O **E1** completa as duas sequências CTC que faltavam para a curva do H9c em
cpu-used=0, levando a Conclusão 2 da tese de 6/8 para **8/8**. O **E4** repete o ponto
`h9c_tau90` com o estudante H9a **neutralizado**, em três sequências além do Neon1224,
para verificar se a decomposição do confound generaliza.

**Scripts:** `src/scripts/fase6/encode_h9c_cq20.py` (E1, novo filtro `--taus`) ·
`encode_h9c_iso.py` (E4, idem) · `report_bloco7.py` (análise)
**Dados:** `results/benchmark/fase6/raw_results.csv` (não versionado), mesmo anchor da Fase 6
**Protocolo:** AOM-CTC All-Intra, 4K 10-bit, cpu-used=0, 15 quadros, CQ {20,32,43,55},
`--tile-columns=1 --threads=2 --row-mt=0`. Execução **estritamente serial** (1 encode por vez)
para preservar a comparabilidade dos tempos com os anchors já medidos.

---

## 1. E1 — o extremo de baixo BD, agora em 8/8 sequências

Faltavam Crosswalk e NocturneDance (as demais 6 já tinham `h9c_tau90/95`). Com os 16
encodes do E1, a curva fecha sobre o conjunto CTC inteiro.

| configuração | BD-rate (%) | TS (%) | speedup | n |
|---|--:|--:|--:|--:|
| **`h9c_tau95`** | **+0,160** | 12,6 | 1,15× | 8 |
| **`h9c_tau90`** | **+0,171** | 13,6 | 1,16× | 8 |
| H9a balanceado (`ml_balanced`) | +0,568 | 17,7 | 1,22× | 8 |
| H9a + H9d (`ml_bal_h9d`) | +0,586 | 18,7 | 1,24× | 8 |
| libaom `cpu-used=1` | +0,449 | 32,6 | 1,51× | 8 |

**A Conclusão 2 sustenta-se sobre as 8 sequências.** O ponto `h9c_tau90/95` é o extremo de
**baixo BD-rate** de toda a família medida: +0,16–0,17% em média, **menos de um terço** do
custo do ponto H9a implantado (+0,57%), ainda entregando 12,6–13,6% de economia de tempo.
As duas sequências novas não só não contradizem como reforçam o padrão — Crosswalk
(`h9c_tau90`: **+0,018%** de BD a 18,2% de TS) e NocturneDance (`h9c_tau95`: **+0,006%**)
são praticamente speedup gratuito, os dois menores custos de BD de toda a tabela.

Por sequência em `report_bloco7.py`; nenhuma das 8 excede +0,27% de BD-rate.

## 2. E4 — quanto do "speedup do H9c" era, na verdade, o H9a

**O problema.** As medições `h9c_tau{N}` setavam apenas as env do H9c e deixavam o estudante
H9a nos seus **defaults compilados** (`tau_none=tau_split=0,9`). Mediam, portanto,
**H9a@0,9 + H9c empilhados** — não o H9c. As linhas `h9ciso_tau90` repetem o mesmo ponto com
o H9a neutralizado (`tau_none=tau_split=2`, que nunca dispara pois as probabilidades do
softmax são ≤1; `tau_rest=−1`, já inerte), isolando o H9c. A única variável alterada é a
neutralização do H9a.

| sequência | `h9c_tau90` (H9a ativo) | `h9ciso_tau90` (H9a inerte) | Δ TS | atribuível ao H9a |
|---|--:|--:|--:|--:|
| Neon1224 | +0,270% / 17,1% | +0,037% / 4,2% | 12,9 pp | **75%** |
| PierSeaSide | +0,177% / 12,8% | −0,021% / 5,4% | 7,5 pp | **58%** |
| Tango | +0,230% / 17,1% | +0,145% / 12,3% | 4,8 pp | **28%** |
| TimeLapse | +0,181% / 11,5% | −0,024% / 0,6% | 10,9 pp | **95%** |
| **média** | **+0,215% / 14,6%** | **+0,034% / 5,6%** | **9,0 pp** | **64%** |

### 2.1 A decomposição generaliza — e o efeito é grande
Em média, **64% da economia de tempo atribuída ao H9c nunca foi do H9c**: era o H9a rodando
nos defaults compilados. O H9c sozinho entrega **5,6% de TS**, não 14,6%. O padrão que até
aqui repousava numa única sequência (Neon1224, 75%) replica-se nas três novas — era
exatamente a fragilidade que o E4 existia para cobrir.

### 2.2 Mas é fortemente dependente de conteúdo
A dispersão é grande e informativa: de **28%** (Tango) a **95%** (TimeLapse). No TimeLapse o
H9c isolado entrega **0,6% de TS** — praticamente nada; toda a economia observada era do H9a.
No Tango o H9c tem substância própria (12,3%). Ou seja, não se trata de um viés constante que
se pudesse descontar com um fator: a contribuição real do H9c varia de desprezível a
moderada conforme o conteúdo.

Nota lateral: nas duas sequências onde o H9c isolado quase não poda (PierSeaSide, TimeLapse)
o BD-rate fica **levemente negativo** (−0,021%, −0,024%). São valores minúsculos, mas exatos
(bytes e PSNR são determinísticos) — podar candidatos altera o contexto de vizinhança das
decisões seguintes, e a busca de partição do AV1 não é monotônica nesse sentido.

## 3. Consequência para a leitura do E1 (honestidade metodológica)

Os dois resultados precisam ser lidos juntos, e a banca fará essa ligação:

**A configuração do §1 é a confundida.** `h9c_tau90/95` = H9a@default + H9c. O extremo de
baixo BD **existe e é um ponto de operação real e implantável** — a tabela do §1 é válida
como caracterização de *ponto de operação*. O que o §2 proíbe é **creditá-lo ao H9c**: em
média só ~36% do seu ganho de tempo vem do H9c.

A redação correta na tese é, portanto: *"a configuração H9a@default + H9c@0,90 é o extremo de
baixo BD-rate da fronteira (+0,17% a 13,6% de TS, 8/8 sequências); a decomposição do E4 mostra
que ~64% dessa economia provém do estudante H9a pré-busca, e não do podador pós-NONE."*

Isso reforça — e agora **quantifica sobre 4 sequências** — a decisão de rejeitar o H9c como
contribuição autônoma. E contrasta com o **H9d**, cujo marginal foi medido desde o início com
a base H9a **fixa e verificada byte-a-byte** (`RESULTADOS_H9d_CTC.md` §1.1): o confound que
inflou o H9c não pode ocorrer ali por construção.

## 4. Limitações
- **15 quadros**, consistente com todas as demais linhas da Fase 6; comparações internas justas.
- **PSNR-Y apenas.**
- **E4 cobre 4 das 8 sequências** (Neon1224 + as 3 novas). A média de 64% é sobre 4, e a
  dispersão 28–95% indica que 4 amostras não fixam bem o valor central — a conclusão robusta
  é a **direção e a ordem de grandeza**, não o número exato.
- **Decomposição de três pernas incompleta.** O terceiro termo (`h9adef` = H9a@default
  sozinho, sem H9c) só existe no **Neon1224**. Com ele nas outras seqs, dava para separar
  H9a puro / H9c puro / interação, em vez de apenas "atribuível ao H9a". 12 encodes (~1h30)
  fechariam isso, se a tese quiser a tabela completa.
- **σ do tempo de parede não medido** (é o E2). Os Δ TS aqui são de 5–13 pp, muito acima do
  piso de ruído estimado (~1–2%), então a conclusão não depende disso.

## 5. Reprodução
```bash
# E1 (16 encodes, ~1h45)
/workspace/build/venv-ml/bin/python src/scripts/fase6/encode_h9c_cq20.py \
    --seqs Crosswalk NocturneDance --cqs 20 32 43 55 --taus 90 95
# E4 (12 encodes, ~1h25)
/workspace/build/venv-ml/bin/python src/scripts/fase6/encode_h9c_iso.py \
    --seqs PierSeaSide Tango TimeLapse --taus 90
# analise
/workspace/build/venv-ml/bin/python src/scripts/fase6/report_bloco7.py
```
