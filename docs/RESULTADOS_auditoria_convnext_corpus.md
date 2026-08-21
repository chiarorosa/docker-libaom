# Auditoria do corpus do ConvNeXt substituto — o braço profundo não era um teste justo

**Data:** 2026-08-20
**Estado:** correção de registro + remedição. Nenhum encode novo; só GPU.
**Pergunta.** A tese sustenta dois enunciados sobre o domínio de pixels a partir do
ConvNeXt substituto: (i) que ele **perde para o `pixels24`**, um MLP sobre 24 atributos
manuais da mesma luminância, e (ii) que **treiná-lo contra a perda de otimalidade piora**
o modelo em toda a faixa, de 1,06× a 3,80×. O segundo enunciado é apresentado em
`RESULTADOS_convnext_regret.md` §1 como ablação controlada, sob a alegação explícita de
que *"o que NÃO muda, deliberadamente: arquitetura, `fusion_dim=128`, formato de saída, e
todo o caminho a jusante"*. Esta auditoria verifica essa alegação nos argumentos gravados
nos próprios checkpoints.

**Veredito:** a alegação **é falsa**. O corpus de treino mudou, e por um fator grande.
Além disso, o braço de entropia cruzada tinha **contaminação da vara held-out**. Os dois
enunciados acima estavam medidos sobre um braço profundo subtreinado, e foram remedidos.

---

## 1. O que os checkpoints declaram

Reprodução da leitura (contêiner):

```bash
build/venv-ml/bin/python - <<'PY'
import torch
for p in ("results/models/surrogate_real/surrogate_best.pt",
          "results/models/surrogate_regret/surrogate_regret_best.pt"):
    print(p, torch.load(p, map_location="cpu", weights_only=False).get("args"))
PY
```

| | `surrogate_real` (`convnext_ce`) | `surrogate_regret` (`convnext_regret`) |
|---|---|---|
| `dataset_dir` | **`results/dataset`** | `results/dataset_h9` |
| `train_seqs` | **`None`** | as 10 canônicas |
| `val_seqs` | **`['Jockey']`** | HoneyBee, FlowerPan, Lips |
| épocas | 30 | 40, paciência 8 |
| lr / decaimento | 1e-3 / 0,01 | 3e-4 / 0,05 |
| lote | 128 | 64 |
| objetivo | CE com rótulo suave sensível a custo, ponderada por frequência inversa, suavização 0,1 | CE ponderada por `1 + α·regret_rel`, α = 3 |
| `fusion_dim` | 128 | 128 |

Mudam sete coisas entre os dois braços, não uma. Atribuir a diferença medida ao
**objetivo** é, portanto, indefensável como está escrito.

## 2. O que era `results/dataset`

Aquele diretório foi descartado e não existe mais em disco, mas a sua composição está
registrada em `RELATORIO_pipeline_dataset_particionamento.md` §8: **quatro sequências**
(Beauty, HoneyBee, Jockey, Bosphorus), **um único ponto de quantização** (`cq-level=32`,
`base_qindex=128`) e **dois quadros** (`[0, 599]`), num total de 817.540 amostras.

A semântica de `data.split_entries` (`data.py:77-93`) é: uma entrada é validação se casar
com algum token de `val_seqs`; caso contrário é treino, **a menos que** `train_seqs` seja
dado. Com `train_seqs=None` e `val_seqs=['Jockey']`, segue que o `surrogate_real` foi
treinado em **Beauty, HoneyBee e Bosphorus**, e selecionado em **Jockey**.

Três consequências, em ordem de gravidade.

**2.1 Contaminação da vara held-out.** A vara do crivo é HoneyBee, FlowerPan, Lips,
Jockey, RaceNight e RiverBank. **Duas** das seis estão comprometidas para este modelo:
HoneyBee entrou no seu treino, e Jockey foi a sequência sobre a qual o seu ponto de
verificação foi selecionado. Um terço da vara não é held-out para ele.

**2.2 Orçamento de treino incomparável.** O `surrogate_real` viu 3 sequências × 1 ponto de
quantização × 2 quadros. Os braços tabulares da tese (`student_h9a`, `pixels24` e a escada
RPP) veem 10 sequências × 4 pontos × 5 quadros, e coletam 120.000 superblocos. A diferença
de cobertura é de ordem de grandeza, e vai toda contra o braço profundo.

**2.3 O canal de quantização foi constante no treino.** `M6_modelos_e_atribuicao.md` §6.1
justifica o plano constante de quantização na entrada dizendo que *"o mesmo conteúdo
apresentado sob forças de quantização distintas produz predições distintas"*. Com um único
`qindex` em todo o treino, esse canal não variou uma vez sequer, e o modelo não teve como
aprender a dependência que a justificativa invoca. O crivo, por outro lado, avalia sobre os
quatro pontos de quantização.

## 3. A remedição — `α = 0` degenera o peso em entropia cruzada pura

`train_surrogate_regret.py:215-236` calcula `w = 1 + α·min(regret_rel/escala, 1)`. Com
`α = 0` o peso é identicamente 1 e a perda vira **entropia cruzada de rótulo duro**, no
mesmo corpus, mesmo split, mesmo escalonamento, mesmo otimizador e mesmo critério de
seleção do braço de `regret`. É a ablação de objetivo que a tese alegou ter feito.

```bash
# braço profundo honesto (objetivo = CE, corpus e split corretos)
build/venv-ml/bin/python src/scripts/partition_model/train_surrogate_regret.py \
  --alpha 0 --fusion-dim 128 --out-dir results/models/surrogate_ce_h9

# controle de capacidade, tudo igual exceto a largura de fusão
build/venv-ml/bin/python src/scripts/partition_model/train_surrogate_regret.py \
  --alpha 0 --fusion-dim 256 --out-dir results/models/surrogate_ce_h9_f256
```

| braço | corpus | α | fusão | melhor `val_wloss` | época | encerramento |
|---|---|--:|--:|--:|--:|---|
| `surrogate_ce_h9` | `dataset_h9`, split 10/3/3 | 0 | 128 | **0,895092** | 5 | parada antecipada, 8 épocas sem melhora |
| `surrogate_ce_h9_f256` | idem | 0 | 256 | 0,903984 | 6 | parada antecipada, 8 épocas sem melhora |
| `surrogate_regret` | idem | 3 | 128 | 0,9250 | — | (registro anterior) |
| `surrogate_regret` (f256) | idem | 3 | 256 | 0,9235 | — | (registro anterior) |

**Ressalva obrigatória de leitura:** `val_wloss` sob α = 0 é entropia cruzada simples,
e sob α = 3 é entropia cruzada ponderada por perda de otimalidade. **As duas colunas de α
não são comparáveis entre si** — são perdas diferentes. A comparação entre objetivos só é
legítima no crivo, que aplica a mesma métrica aos dois. O que as linhas de mesmo α
comparam legitimamente é a **capacidade**.

## 4. O que fica estabelecido

**4.1 A capacidade não é a restrição, agora medido com o objetivo certo.** Sob entropia
cruzada e corpus correto, dobrar a largura de fusão de 128 para 256 **piora** a perda de
validação em 1,0% (0,895092 → 0,903984). O registro anterior media 0,16% de diferença, mas
o fazia entre dois braços de α = 3. O enunciado sobrevive à remedição, e sai dela mais
forte: a largura maior não apenas deixa de ajudar, ela atrapalha.

**4.2 Os dois braços profundos originais não isolavam o objetivo.** Qualquer leitura de
`RESULTADOS_convnext_regret.md` §2 que atribua a piora de 1,06×–3,80× ao alvo de perda de
otimalidade precisa ser reescrita: aquela medição contrasta corpus, taxa de aprendizado,
decaimento, épocas, lote, parada antecipada e objetivo, simultaneamente.

**4.3 A seleção do ponto de verificação do `surrogate_real` continua correta.** A correção
de registro da §4 daquele documento — mínimo de `val_loss` e máximo de macro-F1 ambos na
época 13, que é a salva — permanece válida. O problema auditado aqui não é a seleção; é o
corpus sobre o qual ela foi feita.

## 5. Limitações desta auditoria

- O `results/dataset` original **não existe mais em disco**. A sua composição é lida de
  `RELATORIO_pipeline_dataset_particionamento.md` §8, não verificada diretamente. O que é
  verificado diretamente são os argumentos gravados no checkpoint e a semântica de
  `split_entries`.
- Os braços novos usam o objetivo do script de `regret` com α = 0, que é entropia cruzada
  **simples**. O objetivo do `surrogate_real` era mais elaborado — rótulos suaves sensíveis
  a custo, ponderação por frequência inversa e suavização de 0,1. Portanto `surrogate_ce_h9`
  não é o `surrogate_real` corrigido; é um braço de entropia cruzada limpo. A comparação
  que ele habilita é CE × regret sob tudo o mais idêntico, e não a reprodução do original.
- `α = 0` e `α = 3` são dois pontos; a curva em α não foi varrida.
- O crivo continua sendo triagem, e não adjudicação: no único par com chão de codificador
  limpo ele diverge do encoder.

## 6. Encaminhamento

Os quatro braços profundos — `convnext_ce` (legado, com a contaminação declarada),
`convnext_ce_h9`, `convnext_ce_h9_f256` e `convnext_regret` — são pontuados na mesma vara
pela fronteira única de `results/models/oracle_regret_rpp/`, junto da variância em grade
densa e dos doze bundles da escada RPP. Os documentos da tese que citam a hierarquia do
domínio de pixels passam a sair dessa execução.

> **Procedência.** Argumentos: `results/models/surrogate_real/surrogate_best.pt` e
> `results/models/surrogate_regret/surrogate_regret_best.pt` (chave `args`). Semântica do
> split: `src/scripts/partition_model/data.py:77-93`. Composição do corpus antigo:
> `docs/RELATORIO_pipeline_dataset_particionamento.md` §8. Degeneração do peso:
> `src/scripts/partition_model/train_surrogate_regret.py:215-236`. Registros de treino:
> `results/models/surrogate_ce_h9_train.log` e
> `results/models/surrogate_ce_h9_f256_train.log`. Alegação auditada:
> `docs/RESULTADOS_convnext_regret.md` §1 e §2.
