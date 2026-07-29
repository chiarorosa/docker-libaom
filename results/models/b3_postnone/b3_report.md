# B3 -- HORZ vs VERT: existe sinal direcional?

Experimento off-line, autocontido (`b3_horz_vert.py`), nao faz parte do pipeline implantado. Colapso de rotulos 10->4 (NONE/SPLIT/HORZ/VERT); MLP 36->64->32->4 por nivel, CE de rotulo duro, sem ponderacao de classe, treinado nas mesmas 10 sequencias de treino do estudante H9a implantado.


Pergunta decisiva: condicionado a 'o rotulo verdadeiro e retangular' (HORZ ou VERT), o modelo prediz a DIRECAO acima do acaso (50%)? Acuracia direcional condicional = argmax restrito as colunas HORZ/VERT (softmax completo, ignorando NONE/SPLIT), medida apenas nos nos cujo rotulo verdadeiro e HORZ ou VERT.


## Acuracia direcional por nivel (held-out)

| nivel (px) | n (HORZ|VERT) | prop. HORZ real | acuracia direcional | baseline 50% | baseline 'sempre HORZ' |
|---|---|---|---|---|---|
| 64 | 2707 | 53.5% | 71.9% | 50.0% | 53.5% |
| 32 | 38237 | 51.9% | 68.8% | 50.0% | 51.9% |
| 16 | 150965 | 52.6% | 69.7% | 50.0% | 52.6% |
| **agregado** | 191909 | 52.4% | 69.5% | 50.0% | 52.4% |

## Metricas secundarias por nivel


### nivel 64px

n=48000, macro-F1(4cls)=0.4817, recall(HORZ)=5.5%, recall(VERT)=2.5%

Matriz de confusao (linhas=verdade, colunas=predito; ordem ['NONE', 'SPLIT', 'HORZ', 'VERT']):

```
              NONE     SPLIT      HORZ      VERT
NONE         11852      1850        20        22
SPLIT         1400     30125        13        11
HORZ           339      1027        80         2
VERT           286       933         9        31
```
F1 por classe: NONE=0.8582, SPLIT=0.9201, HORZ=0.1019, VERT=0.0468

### nivel 32px

n=172627, macro-F1(4cls)=0.5939, recall(HORZ)=30.8%, recall(VERT)=25.8%

Matriz de confusao (linhas=verdade, colunas=predito; ordem ['NONE', 'SPLIT', 'HORZ', 'VERT']):

```
              NONE     SPLIT      HORZ      VERT
NONE         66946      3138      1175      1667
SPLIT         4160     51924      3552      1828
HORZ          6365      5067      6121      2291
VERT          5680      6146      1820      4747
```
F1 por classe: NONE=0.8579, SPLIT=0.8130, HORZ=0.3765, VERT=0.3282

### nivel 16px

n=572213, macro-F1(4cls)=0.6235, recall(HORZ)=47.9%, recall(VERT)=46.9%

Matriz de confusao (linhas=verdade, colunas=predito; ordem ['NONE', 'SPLIT', 'HORZ', 'VERT']):

```
              NONE     SPLIT      HORZ      VERT
NONE        375035       171      7376     12372
SPLIT          940     13178      5911      6265
HORZ         21082      4194     38015     16056
VERT         22587      3145     12279     33607
```
F1 por classe: NONE=0.9208, SPLIT=0.5610, HORZ=0.5319, VERT=0.4804

## Veredito

**POSITIVO.** A acuracia direcional condicional supera claramente tanto o acaso (50%) quanto o baseline de 'sempre escolher a classe majoritaria' em todos os niveis -- ha sinal direcional aproveitavel nas features H9a. B3 e viavel e justificaria separar prune_rect_part[HORZ] de prune_rect_part[VERT] em partition_strategy.c:1260-1261.
