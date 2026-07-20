# B3 -- HORZ vs VERT: existe sinal direcional?

Experimento off-line, autocontido (`b3_horz_vert.py`), nao faz parte do pipeline implantado. Colapso de rotulos 10->4 (NONE/SPLIT/HORZ/VERT); MLP 36->64->32->4 por nivel, CE de rotulo duro, sem ponderacao de classe, treinado nas mesmas 10 sequencias de treino do estudante H9a implantado.


Pergunta decisiva: condicionado a 'o rotulo verdadeiro e retangular' (HORZ ou VERT), o modelo prediz a DIRECAO acima do acaso (50%)? Acuracia direcional condicional = argmax restrito as colunas HORZ/VERT (softmax completo, ignorando NONE/SPLIT), medida apenas nos nos cujo rotulo verdadeiro e HORZ ou VERT.


## Acuracia direcional por nivel (held-out)

| nivel (px) | n (HORZ|VERT) | prop. HORZ real | acuracia direcional | baseline 50% | baseline 'sempre HORZ' |
|---|---|---|---|---|---|
| 64 | 2707 | 53.5% | 72.3% | 50.0% | 53.5% |
| 32 | 38237 | 51.9% | 68.5% | 50.0% | 51.9% |
| 16 | 150965 | 52.6% | 69.1% | 50.0% | 52.6% |
| **agregado** | 191909 | 52.4% | 69.1% | 50.0% | 52.4% |

## Metricas secundarias por nivel


### nivel 64px

n=48000, macro-F1(4cls)=0.4500, recall(HORZ)=4.7%, recall(VERT)=2.9%

Matriz de confusao (linhas=verdade, colunas=predito; ordem ['NONE', 'SPLIT', 'HORZ', 'VERT']):

```
              NONE     SPLIT      HORZ      VERT
NONE          9907      3795        16        26
SPLIT         1588     29934         7        20
HORZ           269      1108        68         3
VERT           213       997        13        36
```
F1 por classe: NONE=0.7703, SPLIT=0.8885, HORZ=0.0876, VERT=0.0536

### nivel 32px

n=172627, macro-F1(4cls)=0.5493, recall(HORZ)=26.7%, recall(VERT)=20.2%

Matriz de confusao (linhas=verdade, colunas=predito; ordem ['NONE', 'SPLIT', 'HORZ', 'VERT']):

```
              NONE     SPLIT      HORZ      VERT
NONE         62734      7198      1631      1363
SPLIT         6489     50748      2643      1584
HORZ          7440      5426      5303      1675
VERT          7090      6182      1409      3712
```
F1 por classe: NONE=0.8008, SPLIT=0.7747, HORZ=0.3440, VERT=0.2778

### nivel 16px

n=572213, macro-F1(4cls)=0.5355, recall(HORZ)=42.9%, recall(VERT)=41.5%

Matriz de confusao (linhas=verdade, colunas=predito; ordem ['NONE', 'SPLIT', 'HORZ', 'VERT']):

```
              NONE     SPLIT      HORZ      VERT
NONE        371269      1491     10543     11651
SPLIT         8721      6066      4972      6535
HORZ         28226      2325     34069     14727
VERT         29511      1978     10384     29745
```
F1 por classe: NONE=0.8917, SPLIT=0.3180, HORZ=0.4891, VERT=0.4430

## Veredito

**POSITIVO.** A acuracia direcional condicional supera claramente tanto o acaso (50%) quanto o baseline de 'sempre escolher a classe majoritaria' em todos os niveis -- ha sinal direcional aproveitavel nas features H9a. B3 e viavel e justificaria separar prune_rect_part[HORZ] de prune_rect_part[VERT] em partition_strategy.c:1260-1261.
