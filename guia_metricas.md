# Interpretação dos resultados do treinamento federado

## 1. Organização dos dados e avaliação

O treinamento utiliza o conjunto de dados UCI HAR, dividido oficialmente em dados de treinamento e teste. Cada cliente representa um participante diferente do conjunto de treinamento.

Os dados de cada participante são divididos em duas partes: uma utilizada para o treinamento local do modelo e outra reservada para validação local. Essa divisão permite acompanhar o desempenho de cada cliente sem utilizar os dados de teste oficiais durante o treinamento federado.

A avaliação global é realizada pelo servidor utilizando o conjunto de teste oficial do UCI HAR, composto por participantes que não fazem parte do conjunto de treinamento.

Essa separação permite observar duas perspectivas diferentes:

- **Avaliação dos clientes:** mede o desempenho do modelo nos dados de validação dos participantes utilizados no treinamento federado.
- **Avaliação global:** mede o desempenho do modelo nos dados de teste oficiais, que não foram utilizados no treinamento local.

## 2. Métricas de treinamento local

Durante cada rodada, os clientes recebem o modelo global, realizam o treinamento com seus respectivos dados locais e retornam os resultados ao servidor.

A principal métrica apresentada nessa etapa é:

| Métrica | Significado |
|---|---|
| `train_loss` | Valor da função de perda durante o treinamento local. |

A função de perda indica a diferença entre as previsões do modelo e os rótulos esperados. Valores menores indicam que as previsões estão mais próximas dos valores reais, de acordo com a função de perda utilizada.

O valor apresentado nos resultados agregados corresponde à agregação das métricas retornadas pelos clientes. Portanto, não representa necessariamente a perda de um único participante.

A redução da `train_loss` ao longo das rodadas indica que o modelo está ajustando seus parâmetros durante o treinamento. Entretanto, a perda de treinamento, isoladamente, não é suficiente para determinar a capacidade de generalização do modelo. Para isso, também é necessário observar as métricas de validação e de avaliação global.

## 3. Métricas de validação dos clientes

Após o treinamento local, cada cliente avalia o modelo utilizando sua própria partição de validação.

As métricas retornadas pelos clientes permitem acompanhar o desempenho do modelo nos dados de cada participante e identificar possíveis diferenças entre eles.

| Métrica | Significado |
|---|---|
| `eval_loss` | Valor da função de perda durante a avaliação local. |
| `eval_acc` | Proporção de previsões corretas em relação ao total de exemplos avaliados. |
| `eval_precision` | Proporção de previsões positivas corretas entre todas as previsões positivas. |
| `eval_recall` | Proporção de exemplos positivos identificados corretamente entre todos os exemplos positivos reais. |
| `eval_f1` | Média harmônica entre precisão e revocação. |
| `eval_confidence` | Média da maior probabilidade prevista pelo modelo para cada exemplo avaliado. |
| `eval_correct_confidence` | Confiança média nas previsões classificadas corretamente. |
| `eval_wrong_confidence` | Confiança média nas previsões classificadas incorretamente. |

### 3.1. Acurácia

A acurácia representa a proporção de classificações corretas:

$$
Accuracy = \frac{\text{Previsões corretas}}{\text{Total de exemplos}}
$$

Por exemplo, uma acurácia de 88% significa que o modelo classificou corretamente 88 de cada 100 exemplos avaliados.

### 3.2. Precisão

A precisão indica a proporção de previsões positivas que estavam corretas:

$$
Precision = \frac{TP}{TP + FP}
$$

Em que:

- $TP$: verdadeiros positivos;
- $FP$: falsos positivos.

Uma precisão elevada indica que o modelo produz poucas previsões positivas incorretas.

### 3.3. Revocação

A revocação indica a proporção de exemplos positivos reais que foram identificados corretamente:

$$
Recall = \frac{TP}{TP + FN}
$$

Em que:

- $TP$: verdadeiros positivos;
- $FN$: falsos negativos.

Uma revocação elevada indica que o modelo consegue identificar uma parcela maior dos exemplos pertencentes à classe avaliada.

### 3.4. F1-score

O F1-score combina precisão e revocação por meio da média harmônica:

$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$

Essa métrica permite avaliar conjuntamente a capacidade do modelo de produzir previsões positivas corretas e identificar os exemplos positivos reais.

### 3.5. Confiança das previsões

As métricas de confiança são calculadas a partir das probabilidades produzidas pelo modelo após a aplicação da função softmax.

Para cada exemplo, considera-se a maior probabilidade prevista entre as classes. As métricas representam:

- `eval_confidence`: média da maior probabilidade prevista para todos os exemplos;
- `eval_correct_confidence`: média da maior probabilidade dos exemplos classificados corretamente;
- `eval_wrong_confidence`: média da maior probabilidade dos exemplos classificados incorretamente.

A confiança representa o grau de certeza atribuído pelo modelo à sua previsão, não a probabilidade de que essa previsão esteja efetivamente correta. Por isso, uma confiança elevada não garante uma classificação correta.

A comparação entre `eval_correct_confidence` e `eval_wrong_confidence` permite observar se o modelo tende a atribuir maior confiança às classificações corretas do que às incorretas.

## 4. Comparação entre clientes

Além das métricas agregadas, os resultados podem ser analisados individualmente para identificar diferenças de desempenho entre os participantes.

A comparação da acurácia de cada cliente permite observar a distribuição do desempenho do modelo nos diferentes conjuntos de validação.

Na execução com 10 clientes e 25 rodadas, os resultados da última rodada foram:

| Indicador | Resultado |
|---|---:|
| Acurácia média dos clientes | 88,85% |
| Menor acurácia | 75,00% |
| Cliente com menor acurácia | Client 07 |
| Maior acurácia | 98,55% |
| Cliente com maior acurácia | Client 04 |
| Diferença entre a maior e a menor acurácia | 23,55 pontos percentuais |

A acurácia média apresentada nessa comparação é calculada pela média aritmética das acurácias individuais dos clientes. Cada cliente possui o mesmo peso nesse cálculo, independentemente da quantidade de exemplos em sua partição de validação.

Já o valor agregado de `eval_acc` considera a quantidade de exemplos retornada por cada cliente como peso da agregação. Por esse motivo, os dois resultados podem apresentar pequenas diferenças.

Na execução apresentada, a média aritmética das acurácias individuais foi de 88,85%, enquanto a métrica agregada `eval_acc` foi de 88,82%.

A diferença entre o melhor e o pior resultado também é relevante. Uma variação elevada pode indicar que o desempenho do modelo não é uniforme entre os participantes, o que pode estar relacionado às diferenças na distribuição dos dados locais.

## 5. Avaliação global no servidor

Ao final de cada rodada, o servidor pode avaliar o modelo global utilizando o conjunto de teste oficial do UCI HAR.

Essa avaliação permite acompanhar o desempenho do modelo agregado em dados que não foram utilizados pelos clientes durante o treinamento.

As métricas apresentadas nessa etapa são:

| Métrica | Significado |
|---|---|
| `loss` | Valor da função de perda no conjunto de teste global. |
| `accuracy` | Proporção de classificações corretas no conjunto de teste. |
| `precision` | Precisão macro calculada considerando as classes individualmente. |
| `recall` | Revocação macro calculada considerando as classes individualmente. |
| `f1` | F1-score macro calculado considerando as classes individualmente. |
| `confidence` | Média da maior probabilidade prevista para os exemplos do conjunto de teste. |
| `correct_confidence` | Confiança média das previsões corretas. |
| `wrong_confidence` | Confiança média das previsões incorretas. |

### 5.1. Métricas macro

As métricas `precision`, `recall` e `f1` são calculadas individualmente para cada classe e, em seguida, recebem uma média aritmética.

Para seis classes, a precisão macro é:

$$
Precision_{macro} =
\frac{Precision_1 + Precision_2 + \cdots + Precision_6}{6}
$$

O mesmo procedimento é utilizado para a revocação e o F1-score.

Dessa forma, todas as classes possuem o mesmo peso no cálculo da métrica, independentemente da quantidade de exemplos pertencentes a cada uma.

Essa abordagem permite avaliar o desempenho do modelo considerando todas as classes, evitando que classes com maior quantidade de exemplos dominem o resultado.

### 5.2. Resultados da avaliação global

Na execução com 10 clientes e 25 rodadas, os resultados da última rodada foram:

| Métrica | Resultado |
|---|---:|
| `loss` | 0,3794 |
| `accuracy` | 86,26% |
| `precision` | 0,8721 |
| `recall` | 0,8561 |
| `f1` | 0,8584 |
| `confidence` | 85,59% |
| `correct_confidence` | 88,43% |
| `wrong_confidence` | 67,72% |

A acurácia global de 86,26% indica que o modelo classificou corretamente aproximadamente 86 de cada 100 exemplos do conjunto de teste.

O F1-score macro de 0,8584 representa o equilíbrio entre precisão e revocação, considerando as seis classes com o mesmo peso.

A confiança média das previsões corretas foi de 88,43%, enquanto a das previsões incorretas foi de 67,72%. Essa diferença indica que, nessa avaliação, o modelo atribuiu maior confiança às classificações corretas.

## 6. Resultados apresentados pelo Flower

Ao finalizar a execução, o Flower apresenta uma seção denominada `Final results`, que reúne as métricas registradas durante o treinamento e a avaliação.

Essa seção é dividida em diferentes grupos.

### 6.1. Aggregated ClientApp-side Train Metrics

Contém as métricas retornadas pelos clientes durante o treinamento local.

No projeto, a principal métrica registrada nesse grupo é:

- `train_loss`: perda agregada do treinamento local em cada rodada.

A comparação dos valores entre as rodadas permite acompanhar a evolução da função de perda durante o treinamento federado.

### 6.2. Aggregated ClientApp-side Evaluate Metrics

Contém as métricas retornadas pelos clientes durante a avaliação local.

No projeto, são apresentadas:

- `eval_loss`;
- `eval_acc`;
- `eval_precision`;
- `eval_recall`;
- `eval_f1`;
- `eval_confidence`;
- `eval_correct_confidence`;
- `eval_wrong_confidence`.

Esses valores representam a agregação das avaliações realizadas nas partições de validação dos clientes.

As métricas são agregadas considerando a quantidade de exemplos informada por cada cliente. Portanto, as métricas agregadas de precisão, revocação e F1-score correspondem à agregação das métricas calculadas localmente, e não ao cálculo direto dessas métricas sobre todas as previsões individuais reunidas em um único conjunto.

### 6.3. ServerApp-side Evaluate Metrics

Contém as métricas calculadas pelo servidor durante a avaliação do modelo global no conjunto de teste oficial.

No projeto, são apresentadas:

- `accuracy`;
- `loss`;
- `precision`;
- `recall`;
- `f1`;
- `confidence`;
- `correct_confidence`;
- `wrong_confidence`.

Esses resultados permitem acompanhar a evolução do modelo global ao longo das rodadas e verificar seu desempenho nos dados de teste.

A entrada identificada como rodada 0 corresponde à avaliação inicial do modelo global, antes das atualizações realizadas pelo treinamento federado.

### 6.4. Notação científica

Alguns valores podem aparecer na notação científica. Por exemplo:

| Notação | Valor decimal |
|---|---:|
| `8.6257e-01` | 0,86257 |
| `3.7945e-01` | 0,37945 |
| `8.5836e-01` | 0,85836 |

Para converter um resultado em percentual, basta multiplicar o valor decimal por 100. Assim, `8.6257e-01` corresponde a aproximadamente 86,26%.

## 7. Considerações sobre a interpretação dos resultados

As métricas de treinamento, validação local e avaliação global representam etapas diferentes do processo e não devem ser interpretadas como se fossem calculadas sobre o mesmo conjunto de dados.

A perda de treinamento indica o comportamento do modelo durante os ajustes realizados pelos clientes. As métricas de validação permitem observar seu desempenho nos dados locais reservados para avaliação. Já as métricas do servidor indicam o desempenho do modelo global no conjunto de teste oficial.

Também é importante considerar que a agregação das métricas locais não equivale necessariamente ao cálculo das mesmas métricas sobre todas as previsões reunidas. Isso ocorre especialmente com precisão, revocação e F1-score, que são calculados localmente e depois agregados.

Por fim, a implementação de `test_metrics` calcula a perda dividindo a soma das perdas dos lotes pela quantidade de lotes avaliados. Portanto, cada lote possui o mesmo peso nesse cálculo, mesmo quando o último lote contém menos exemplos. A acurácia, por sua vez, é calculada dividindo a quantidade de previsões corretas pelo total de exemplos avaliados.