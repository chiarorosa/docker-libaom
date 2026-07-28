# Decisões de escopo — o que a tese deliberadamente não faz

**Data:** 2026-07-26
**Propósito.** Distinguir, no capítulo de metodologia, o que é **limitação** (algo que
enfraquece uma conclusão) do que é **decisão de escopo** (algo deliberadamente fora do
recorte, com justificativa). Itens registrados aqui foram avaliados e **fechados**; não são
pendências.

---

## 1. Métrica: PSNR-Y apenas

**Decisão:** todas as taxas BD da tese são calculadas sobre **PSNR-Y**, sem PSNR-U/V,
SSIM/MS-SSIM, VMAF, PSNR-HVS-M ou CIEDE2000.

**Justificativa:** é a convenção dominante na literatura de decisão de particionamento em
codificadores, que é onde a contribuição se situa — a comparabilidade com o estado da arte
da área pesa mais do que a cobertura métrica. A luminância também é o plano sobre o qual a
decisão de partição de fato opera: o estudante consome atributos de luminância
(`student_node_features`), e a partição do croma é derivada, não decidida de forma
independente.

**Divergência declarada:** a CTC §2.2 arrola um conjunto métrico mais amplo. Esta tese
reporta um subconjunto dele. Isso **não invalida** nenhuma comparação interna — todas as
configurações são medidas com a mesma métrica contra o mesmo anchor — mas significa que os
números não são diretamente comparáveis a submissões CTC que reportem APSNR-YUV.

## 2. Quinze quadros — conformidade, não redução

**Decisão:** 15 quadros por sequência em toda a Fase 6.

**Justificativa:** é **exatamente o que a CTC especifica** para All Intra. A §4.1 diz *"the
test following this configuration uses the first 15 frames"* e *"for video test data (Class
A, Class B and Class G), `--limit=15` should be used"*. O changelog da v7.0 registra a
mudança deliberada *"AI frame count: 30 → 15 frames (`--limit=15`)"*.

**Correção de registro:** documentos anteriores desta tese descreviam os 15 quadros como um
recorte próprio ("não os 60 do CTC pleno"). Isso estava **errado** e foi corrigido em
`RESULTADOS_H9d_CTC.md` e `RESULTADOS_BLOCO7_E1_E4.md`. Não há déficit de quadros a defender.

## 3. Divergências forçadas pela versão do libaom

Auditando o comando de encode contra a CTC §4.1, duas diferenças são **impostas pelo
codificador**, não escolhidas:

| CTC §4.1 | esta tese | motivo |
|---|---|---|
| `--qp=x` | `--cq-level=x` | `aomenc` do libaom v3.10.0 **não expõe** `--qp`; a única escala de quantização disponível é `--cq-level` |
| `--use-fixed-qp-offsets=1` | (ausente) | flag **inexistente** neste build (verificado em `aomenc --help`) |

O texto da CTC v9 tem como alvo a cadeia de ferramentas do AV2/AVM; esta tese avalia o
**libaom AV1**, cuja interface é anterior. Todos os demais parâmetros da §4.1 são
reproduzidos literalmente (`--cpu-used=0 --passes=1 --end-usage=q --kf-min-dist=0
--kf-max-dist=0 --deltaq-mode=0 --enable-tpl-model=0 --enable-keyframe-filtering=0 --obu`),
mais a regra de ladrilhamento 4K da §4 para a Classe A1 (`--tile-columns=1 --tile-rows=0
--threads=2 --row-mt=0`).

Impacto sobre as conclusões: **nenhum**. Em All Intra com `--kf-max-dist=0` todo quadro é
quadro-chave, de modo que deslocamentos fixos de QP entre tipos de quadro não teriam efeito;
e `--cq-level` percorre a mesma faixa de quantização, com a grade 20/32/43/55 aplicada de
forma idêntica a **todas** as configurações, inclusive o anchor.

## 4. B2 / E6 — τ adaptativo por qindex: não será levado ao encoder

**Decisão:** a calibração offline fica registrada
(`RESULTADOS_modelagem_B2_tau_qindex.md`); a confirmação no encoder (E6) **não será
executada**.

**Justificativa:** a própria calibração classificou o resultado como **positivo, porém
imaterial** — o ganho a custo casado é pequeno demais para deslocar a fronteira BD×tempo de
forma mensurável acima do piso de ruído do tempo de parede. Gastar ~3h de encodes para
confirmar um efeito que se sabe menor que a resolução do experimento não produziria número
de tabela defensável. O achado permanece como registro de calibração.

## 5. `h9acomb` — hipótese suficientemente testada

**Decisão:** a campanha `h9acomb` (~168 encodes) **não será completada**.

**Justificativa:** trata-se de consolidar um **resultado negativo já visível** nos dados
parciais (notadamente no Tango). O plano de ações já a classificava como "não priorizar". A
hipótese foi testada o bastante para ser reportada como negativa; encodes adicionais
aumentariam a precisão de um número que não muda conclusão alguma.

---

## O que continua genuinamente em aberto

Para não confundir decisão com pendência, os itens ainda abertos são:

| # | item | natureza | estado |
|---|---|---|---|
| ~~1~~ | ~~**B3** — sinal direcional (HORZ vs VERT) confirmado offline, nunca implantado em C~~ | possível terceira solução positiva | **FECHADO 26/07 — negativo.** Acurácia direcional plana (69,2%→69,5% em controle pareado); recall ~42%→~47%, insuficiente. Nunca chegou a C. `RESULTADOS_modelagem_B3_horz_vert.md §7` |
| ~~2~~ | ~~**ConvNeXt** — retreino com alvo de *regret*; o teto de pixels foi medido com modelo sobreajustado e objetivo errado~~ | pode mudar uma conclusão | **FECHADO 26/07 — refutado.** O retreino piorou o modelo em toda a faixa (1,06–3,80×). A premissa do "sobreajuste" era **falsa** (§4 do doc). Achado maior: o ConvNeXt perde para o `pixels24` e **não é teto de pixels**. `RESULTADOS_convnext_regret.md` |
| ~~3~~ | ~~**Fronteira do H9d** — o capítulo de resultados traz um único ponto de operação (PL10 sobre P_rect), enquanto o H9a traz uma curva~~ | lacuna narrativa nos resultados | **FECHADO 27/07** — 96 encodes, 4 pontos. Os 4 batem o knob de τ; o implantado é o melhor (3,38×). Achado novo: **inerte sobre a base agressiva** (+0,17 pp, 1/8 seqs acima da resolução) → a aditividade depende do ponto de operação |
| 4 | **E5** — ablação da CB-1 com 10 quadros nas **3** sequências de validação | blindagem da metodologia | 🔄 **EM EXECUÇÃO desde 27/07** |

O **E5 está pausado por decisão**, não descartado: parte do que ele responderia é
subsumida pelo item 2 (que testa o **modelo-teto** com o objetivo certo, enquanto o
E5 testa apenas o estudante). A submissão fica a decidir depois do item 2.

> **Atualização (27/07) — a condição que adiava o E5 já ocorreu, e o resultado aumentou o
> valor dele.** O item 2 fechou, mas **não** subsumiu o E5: descobriu-se que o `ml` da
> ablação de atribuição usa as **36 features do H9a**, não um estudante de pixels
> (`partition_strategy.c:2164`), de modo que o E5 mede a atribuição do **podador
> implantado** — não do teto. E como o capítulo passou a apoiar-se na hierarquia do crivo
> A5, que é declaradamente **não-adjudicante**, a confirmação no codificador ficou mais
> necessária, não menos.
>
> **Decisão tomada e campanha lançada (27/07 12:54 UTC):** as sequências de validação
> — o papel delas na tese é exatamente este.
>
> **Escopo fixado em 2 das 3 (28/07).** O portão é o `ml` dominar a variância a tempo
> casado em **≥2 das 3** seqs, e **FlowerPan + Lips** já o decidem. A **HoneyBee fica
> de fora por decisão**: é a sequência em que o grid de τ original foi calibrado,
> portanto a menos independente das três, e custaria ~13 h para o menor acréscimo
> possível. Registre-se como decisão de escopo — não é sequência que falhou nem
> resultado omitido.
>
> **Custo real:** ~44 min por ponto de operação (4 CQ), medido na FlowerPan. As duas
> estimativas anteriores (~12 h para 3 seqs, depois ~18 h) eram otimistas; a
> extrapolação partia do Jockey, e a FlowerPan é 4K a 50 fps, com âncora de 1204 s em
> cq20 contra 887 s do Jockey. Ver `ANDAMENTO_tese.md §0.2`.

Ver `ANDAMENTO_tese.md §0` para a fila corrente e as restrições de agenda.

Ver `PLANO_ACOES_EXPLORACAO_OPORTUNIDADES.md` (fora do repositório) para o histórico completo.
