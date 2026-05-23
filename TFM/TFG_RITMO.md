<img src="./media/image1.png" style="width:1.99306in;height:0.86806in" /> <img src="./media/image2.png" style="width:1.55556in;height:0.63889in" />

Tokenización de series temporales mediante estados ocultos de Markov: comparativa con técnicas determinísticas para predicción a largo plazo

TIME SERIES TOKENIZATION VIA HIDDEN MARKOV MODELS: A COMPARATIVE STUDY WITH DETERMINISTIC TECHNIQUES FOR LONG-TERM FORECASTING

Alumno: Jaime Oriol Goicoechea

Tutor: Julio Emilio Sandubete Galán

Email: joriolgo@gmail.com

Grado en Business Analytics

Curso académico 2025-2026

**Resumen**

Este trabajo propone y evalúa RITMO (*Regímenes latentes mediante Inferencia Temporal con Markov Oculto*), un sistema de tokenización de series temporales basado en Hidden Markov Models que codifica las dinámicas de transición entre regímenes dentro del propio *token* y genera *embeddings* estructurados con dimensiones interpretables. La evaluación combina dos escenarios. El **Plan A** compara seis técnicas —discretización, *text-based*, *patching*, descomposición, *foundation models* y RITMO— sobre un *transformer* común, en cuatro *datasets* y 3 *seeds* {42, 2021, 7}. El **Plan B** enfrenta a RITMO-M (`--features M`) contra DLinear, PatchTST, TimeMixer y TimeXer con sus hiperparámetros publicados. En Plan A, RITMO logra el primer puesto en MSE promedio en Weather y ETTh2, y gana la transferencia *cross-domain* sobre Exchange. En Plan B, RITMO-M es competitivo —tercero en Weather, cuarto en los demás—, supera a DLinear en los cuatro *datasets* y vence a los cuatro *baselines* en ETTh2 al horizonte 720. La ventaja aparece cuando el *dataset* satisface una de dos condiciones empíricas: regímenes persistentes alineados con la escala predictiva, o tolerancia al *distribution shift*. La interpretabilidad directa de los *embeddings* estructurados completa la contribución principal del trabajo.

**Palabras clave:** series temporales; tokenización; Hidden Markov Models; *transformers*; predicción a largo plazo; *embeddings* estructurados

**Abstract**

This work proposes and evaluates RITMO (*Regime Inference via Temporal Markov Observation*), a time-series tokenization system based on Hidden Markov Models that encodes regime-transition dynamics inside the token itself and produces structured embeddings with interpretable dimensions. Evaluation combines two scenarios. **Plan A** compares six techniques —discretization, text-based, patching, decomposition, foundation models and RITMO— on a common transformer across four datasets and 3 seeds {42, 2021, 7}. **Plan B** pits RITMO-M (`--features M`) against DLinear, PatchTST, TimeMixer and TimeXer with their published configurations. In Plan A, RITMO ranks first in average MSE on Weather and ETTh2, and wins the cross-domain transfer on Exchange. In Plan B, RITMO-M is competitive —third on Weather, fourth elsewhere—, beats DLinear on all four datasets, and beats all four baselines on ETTh2 at horizon 720. The advantage appears when the dataset meets one of two empirical conditions: persistent regimes aligned with the predictive horizon, or tolerance to distribution shift. The direct interpretability of the structured embeddings completes the main contribution of the work.

> **Key words:** time series; tokenization; Hidden Markov Models; transformers; long-term forecasting; structured embeddings

**Agradecimientos**

*"Salid y disfrutad."* — Johan Cruyff

Cruyff pronunció esta frase ante sus jugadores momentos antes de la final de Wembley de 1992, posiblemente el partido más importante de la historia del Barça. *Disfrutad*. Ante uno de los mayores retos imaginables, su mensaje no fue de presión ni de táctica, sino de libertad. Precisamente eso es lo que mejor describe lo que he sentido durante este trabajo, que las personas que me rodean me han dejado, en todo momento, *salir y disfrutar* del reto.

A mis padres, por apoyarme siempre en cada decisión que tomo y darme la libertad de elegir ser quien soy.

A mi hermano, Dani, por ser otro pilar imprescindible y depositar su confianza en todas las decisiones que tomo.

A la Dra. Ana Lazcano, por abrirme siempre su puerta, brindarme esta oportunidad e introducirme al fascinante mundo de la investigación.

Al Dr. Julio E. Sandubete, por apoyarme, guiarme y tutorizarme durante este trabajo, dejándome libertad para aprender y metiendo baza cuando tocaba.

Gracias a todos por dejarme *salir y disfrutar*.















**ÍNDICE**

[Índice de Figuras [6](#índice-de-figuras)](#índice-de-figuras)

[Índice de Tablas [8](#índice-de-tablas)](#índice-de-tablas)

[1. INTRODUCCIÓN [1](#introducción)](#introducción)

[1.1. Descripción del problema [1](#descripción-del-problema)](#descripción-del-problema)

[1.2. Motivación [2](#motivación)](#motivación)

[1.3. Organización de la memoria [3](#organización-de-la-memoria)](#organización-de-la-memoria)

[2. OBJETIVOS [4](#objetivos)](#objetivos)

[2.1. Objetivo general [4](#objetivo-general)](#objetivo-general)

[2.2. Objetivos específicos [5](#objetivos-específicos)](#objetivos-específicos)

[3. ESTADO DEL ARTE [6](#estado-del-arte)](#estado-del-arte)

[3.1. LLMs y transformers en series temporales [6](#llms-y-transformers-en-series-temporales)](#llms-y-transformers-en-series-temporales)

[3.2. Técnicas de tokenización [7](#técnicas-de-tokenización)](#técnicas-de-tokenización)

[3.3. Transformers especializados para series temporales [8](#transformers-especializados-para-series-temporales)](#transformers-especializados-para-series-temporales)

[3.4. Hidden Markov Models [9](#hidden-markov-models)](#hidden-markov-models)

[3.5. Normalización y preprocesamiento [10](#normalización-y-preprocesamiento)](#normalización-y-preprocesamiento)

[3.6. Saturación de benchmarks [10](#saturación-de-benchmarks)](#saturación-de-benchmarks)

[3.7. Gap identificado [11](#gap-identificado)](#gap-identificado)

[4. METODOLOGÍA [11](#metodología)](#metodología)

[4.1. Descripción del marco metodológico utilizado [11](#descripción-del-marco-metodológico-utilizado)](#descripción-del-marco-metodológico-utilizado)

[4.2. Marco Teórico [12](#marco-teórico)](#marco-teórico)

[4.2.1. Hidden Markov Models [13](#hidden-markov-models-1)](#hidden-markov-models-1)

[4.2.2. Algoritmo Forward-Backward [13](#algoritmo-forward-backward)](#algoritmo-forward-backward)

[4.2.3. Algoritmo de Baum-Welch [14](#algoritmo-de-baum-welch)](#algoritmo-de-baum-welch)

[4.2.4. Algoritmo de Viterbi [15](#algoritmo-de-viterbi)](#algoritmo-de-viterbi)

[4.2.5. Reversible Instance Normalization [15](#reversible-instance-normalization)](#reversible-instance-normalization)

[4.2.6. Embeddings estructurados [15](#embeddings-estructurados)](#embeddings-estructurados)

[4.2.7. Técnicas de tokenización comparadas [16](#técnicas-de-tokenización-comparadas)](#técnicas-de-tokenización-comparadas)

[4.2.8. Modelo de predicción: Transformer común [22](#modelo-de-predicción-transformer-común)](#modelo-de-predicción-transformer-común)

[4.3. Herramientas y tecnologías [24](#herramientas-y-tecnologías)](#herramientas-y-tecnologías)

[4.4. Datasets e ingeniería del dato [24](#datasets-e-ingeniería-del-dato)](#datasets-e-ingeniería-del-dato)

[4.4.1. Origen, obtención y criterio de selección [25](#origen-obtención-y-criterio-de-selección)](#origen-obtención-y-criterio-de-selección)

[4.4.2. Estructura, validación e integridad [26](#estructura-validación-e-integridad)](#estructura-validación-e-integridad)

[4.4.3. Análisis estadístico y temporal [27](#análisis-estadístico-y-temporal)](#análisis-estadístico-y-temporal)

[4.4.4. Transformaciones de preprocesamiento aplicadas [28](#transformaciones-de-preprocesamiento-aplicadas)](#transformaciones-de-preprocesamiento-aplicadas)

[4.5. Métricas de evaluación [28](#métricas-de-evaluación)](#métricas-de-evaluación)

[4.5.1. Métricas intrínsecas de tokenización [28](#métricas-intrínsecas-de-tokenización)](#métricas-intrínsecas-de-tokenización)

[4.6. Métricas de predicción [30](#métricas-de-predicción)](#métricas-de-predicción)

[4.7. Protocolo multi-seed y selección de K [31](#protocolo-multi-seed-y-selección-de-k)](#protocolo-multi-seed-y-selección-de-k)

[5. DESARROLLO TÉCNICO [32](#desarrollo-técnico)](#desarrollo-técnico)

[5.1. Diseño experimental [33](#diseño-experimental)](#diseño-experimental)

[5.1.1. Plan A: comparación controlada de tokenizaciones [33](#plan-a-comparación-controlada-de-tokenizaciones)](#plan-a-comparación-controlada-de-tokenizaciones)

[5.1.2. Plan B: validación frente al estado del arte [34](#plan-b-validación-frente-al-estado-del-arte)](#plan-b-validación-frente-al-estado-del-arte)

[5.2. Fase 1: Normalización RevIN [35](#fase-1-normalización-revin)](#fase-1-normalización-revin)

[5.3. Fase 2: Entrenamiento HMM (Baum-Welch) [36](#fase-2-entrenamiento-hmm-baum-welch)](#fase-2-entrenamiento-hmm-baum-welch)

[5.4. Fase 3: Tokenización (Viterbi) [37](#fase-3-tokenización-viterbi)](#fase-3-tokenización-viterbi)

[5.4.1. Variantes de tokenización HMM [38](#variantes-de-tokenización-hmm)](#variantes-de-tokenización-hmm)

[5.5. Fase 4: Embeddings estructurados [39](#fase-4-embeddings-estructurados)](#fase-4-embeddings-estructurados)

[5.6. Comparativa intrínseca de técnicas de tokenización [40](#comparativa-intrínseca-de-técnicas-de-tokenización)](#comparativa-intrínseca-de-técnicas-de-tokenización)

[5.7. Fase 5: Modelo de predicción (Transformer) [44](#fase-5-modelo-de-predicción-transformer)](#fase-5-modelo-de-predicción-transformer)

[6. RESULTADOS [46](#_Toc228694377)](#_Toc228694377)

[6.1. Protocolo de ejecución del Plan A [47](#protocolo-de-ejecución-del-plan-a)](#protocolo-de-ejecución-del-plan-a)

[6.2. Resultados Plan A: Grupo 1 (*in-domain*) [47](#resultados-plan-a-grupo-1-in-domain)](#resultados-plan-a-grupo-1-in-domain)

[6.3. Resultados Plan A: Grupo 2 (transferencia *cross-domain* del *tokenizer* HMM) [53](#resultados-plan-a-grupo-2-transferencia-cross-domain-del-tokenizer-hmm)](#resultados-plan-a-grupo-2-transferencia-cross-domain-del-tokenizer-hmm)

[6.4. Interpretabilidad de los regímenes aprendidos [55](#interpretabilidad-de-los-regímenes-aprendidos)](#interpretabilidad-de-los-regímenes-aprendidos)

[6.5. Resultados Plan B: comparación frente al estado del arte [57](#resultados-plan-b-comparación-frente-al-estado-del-arte)](#resultados-plan-b-comparación-frente-al-estado-del-arte)

[7. DISCUSIÓN DE RESULTADOS [61](#discusión-de-resultados)](#discusión-de-resultados)

[7.1. Qué demuestran los resultados [62](#qué-demuestran-los-resultados)](#qué-demuestran-los-resultados)

[7.2. Qué no demuestran los resultados [62](#qué-no-demuestran-los-resultados)](#qué-no-demuestran-los-resultados)

[7.3. Dónde funciona y dónde no funciona RITMO [63](#dónde-funciona-y-dónde-no-funciona-ritmo)](#dónde-funciona-y-dónde-no-funciona-ritmo)

[7.4. Implicaciones para el diseño de tokenizadores probabilísticos [66](#implicaciones-para-el-diseño-de-tokenizadores-probabilísticos)](#implicaciones-para-el-diseño-de-tokenizadores-probabilísticos)

[8. CONCLUSIONES [67](#conclusiones)](#conclusiones)

[8.1. Conclusiones generales [67](#conclusiones-generales)](#conclusiones-generales)

[8.2. Limitaciones [68](#limitaciones)](#limitaciones)

[8.3. Líneas futuras [68](#líneas-futuras)](#líneas-futuras)

[9. REFERENCIAS BIBLIOGRÁFICAS [70](#referencias-bibliográficas)](#referencias-bibliográficas)

[ANEXO A: Análisis exploratorio de los seis *datasets* [75](#anexo-a-análisis-exploratorio-de-los-seis-datasets)](#anexo-a-análisis-exploratorio-de-los-seis-datasets)

[ANEXO B: Barrido de K del Plan A — MSE de validación @ O = 96 sobre 3 *seeds* [78](#_Toc228694394)](#_Toc228694394)

[ANEXO C: Barrido de K del Plan B — MSE de validación promediado sobre 4 horizontes (seed = 2021) [81](#anexo-c-barrido-de-k-del-plan-b-mse-de-validación-promediado-sobre-4-horizontes-seed-2021)](#anexo-c-barrido-de-k-del-plan-b-mse-de-validación-promediado-sobre-4-horizontes-seed-2021)

[ANEXO D: Comparativa visual de predicciones por técnica — Electricity pl = 96 [84](#anexo-d-comparativa-visual-de-predicciones-por-técnica-electricity-pl-96)](#anexo-d-comparativa-visual-de-predicciones-por-técnica-electricity-pl-96)

# ÍNDICE DE FIGURAS 

[Ilustración 1. Pipeline RITMO: cinco etapas del flujo para predicción de series temporales con HMM [12](#_Ref226788084)](#_Ref226788084)

[Ilustración 2. Serie ETTh2 normalizada con RevIN, base común para los ejemplos de las seis técnicas de tokenización. [16](#_Ref226879634)](#_Ref226879634)

[Ilustración 3. Discretización (SAX-inspired) sobre ETTh2: breakpoints gaussianos y asignación de símbolos discretos. [17](#_Ref226879647)](#_Ref226879647)

[Ilustración 4. Tokenización text-based (LLMTime-inspired) sobre ETTh2: serialización decimal carácter a carácter. [17](#_Ref226879662)](#_Ref226879662)

[Ilustración 5. Patching (PatchTST-inspired) sobre ETTh2: segmentación en patches no solapados y estadísticos por patch. [18](#_Ref226879672)](#_Ref226879672)

[Ilustración 6. Descomposición (Autoformer-inspired) sobre ETTh2: separación en componente de tendencia y estacional. [19](#_Ref226879682)](#_Ref226879682)

[Ilustración 7. Tokenización foundation (MOMENT-inspired) sobre ETTh2: enmascaramiento aleatorio de patches y error de reconstrucción. [20](#_Ref226879744)](#_Ref226879744)

[Ilustración 8. Tokenización HMM (K = 5) sobre ETTh2: asignación de estados, secuencia de tokens, matriz de transición y ocupación. [21](#_Toc229829349)](#_Toc229829349)

[Ilustración 9. Flujo de integración del *TransformerCommon* con las seis técnicas de tokenización: diagrama de bloques de los seis pasos del *forward pass* y sus dimensiones tensoriales. [23](#_Ref226962836)](#_Ref226962836)

[Ilustración 10. Validación de RevIN sobre ETTh2: serie original (μ = 28.82, σ = 11.40) y serie tras normalización per-window (μ ≈ 0, σ ≈ 1). [36](#_Toc229829351)](#_Toc229829351)

[Ilustración 11. Curva de convergencia de Baum-Welch sobre ETTh2 (K = 5): log-verosimilitud monótonamente creciente y \|ΔLL\| decreciente exponencialmente hasta el umbral ε = 10⁻⁴. [37](#_Ref226882016)](#_Ref226882016)

[Ilustración 12. Tokenización Viterbi sobre ETTh2 con K = 5 (ejemplo ilustrativo): asignación de estados, secuencia de tokens y distribución de frecuencias. [38](#_Toc229829353)](#_Toc229829353)

[Ilustración 13. Espacio de *embeddings* HMM sobre ETTh2 K = 5 (ejemplo ilustrativo): plano $\mu k - \ \sigma k$ de los cinco regímenes y matriz de transición A asociada. [39](#_Ref226882080)](#_Ref226882080)

[Ilustración 14. Plan A — Curvas MSE frente a horizonte de predicción $O\  \in \text{\{}96,\ 192,\ 336,\ 720\text{\}}$ para los cuatro datasets del Grupo 1 y las seis técnicas comparadas, mean ± std sobre 3 seeds (288 ejecuciones in-domain). [50](#_Toc229829355)](#_Toc229829355)

[Ilustración 15. Plan A — Ranking heatmap de las seis técnicas por avg MSE en cada dataset del Grupo 1 (1 = mejor, 6 = peor; agregado sobre 3 seeds). [52](#_Toc229829356)](#_Toc229829356)

[Ilustración 16. Tokenización HMM sobre Weather (K = 4 soft residual, configuración óptima del Plan A): asignación de estados, secuencia de tokens y distribución de frecuencias. [55](#_Ref228535802)](#_Ref228535802)

[Ilustración 17. HMM Weather (K = 4 soft residual) — izquierda: espacio de embeddings μₖ−σₖ con callouts por estado (μₖ, σₖ, fₖ, A\[k, k\]); derecha: matriz de transición A (media diagonal = 0.94). [56](#_Ref228558221)](#_Ref228558221)

[Ilustración 18. Plan B — MSE por horizonte de predicción $\mathbf{O}\  \in \ \text{\{}\mathbf{96},\ \mathbf{192},\ \mathbf{336},\ \mathbf{720}\text{\}}\ $para los cuatro datasets del Grupo 1 y los cinco modelos comparados (RITMO-M + 4 baselines SOTA), seed única (2021) (320 ejecuciones controladas). [59](#_Ref228557412)](#_Ref228557412)

[Ilustración 19. Plan B — Ranking heatmap de los cinco modelos por avg MSE en cada dataset del Grupo 1 (1 = mejor, 5 = peor; seed única = 2021). [60](#_Ref228557421)](#_Ref228557421)

[Ilustración 20. Series temporales completas de los seis datasets, con los cortes verticales que marcan las particiones train / val / test. [75](#_Toc229829361)](#_Toc229829361)

[Ilustración 21. Histogramas de distribución del target en cada dataset; permiten contrastar visualmente la asimetría, la dispersión y la presencia de colas. [75](#_Toc229829362)](#_Toc229829362)

[Ilustración 22. Funciones de autocorrelación (ACF) hasta lag 400 para los seis datasets; revelan los picos diarios y semanales reportados en la Tabla 4. [76](#_Toc229829363)](#_Toc229829363)

[Ilustración 23. Comparación visual del distribution shift entre train y test; ilustra el desplazamiento de la media y de la dispersión que motiva la doble normalización descrita en la sección 4.4.4. [76](#_Toc229829364)](#_Toc229829364)

[Ilustración 24. Barrido de K — Plan A: MSE de validación @ $O\  = \ 96$ frente a $K\  \in \text{\{}3,\ldots,10\text{\}}$ por dataset y variante HMM, media ± desviación sobre 3 seeds {42, 2021, 7}. Línea roja discontinua: K seleccionado (Tabla 5). [78](#_Ref228604851)](#_Ref228604851)

[Ilustración 25. Barrido de K — Plan B (RITMO-M, `--features M`, seed = 2021): MSE de validación promediado sobre los 4 horizontes $O\  \in \text{\{}96,\ 192,\ 336,\ 720\text{\}}$ frente a $K\  \in \text{\{}3,\ldots,10\text{\}}$ por dataset y variante HMM. Línea roja discontinua: K seleccionado (Tabla 6). [81](#_Ref228606502)](#_Ref228606502)

[Ilustración 26. RITMO (HMM, K=3 soft residual) — predicción sobre Electricity pl = 96 (sample 2043, seed = 2021); MSE de muestra = 27 481.59 ($R2 = 0.894$). [84](#_Toc229829367)](#_Toc229829367)

[Ilustración 27. PatchTST-inspired — predicción sobre Electricity pl = 96 (sample 2043, seed = 2021); MSE de muestra = 160 779.46 ($R2 = 0.381$). [84](#_Toc229829368)](#_Toc229829368)

[Ilustración 28. MOMENT-inspired — predicción sobre Electricity pl = 96 (sample 2043, seed = 2021); MSE de muestra = 126 217.33 ($R2 = 0.514$). [84](#_Toc229829369)](#_Toc229829369)

[Ilustración 29. Autoformer-inspired — predicción sobre Electricity pl = 96 (sample 2043, seed = 2021); MSE de muestra = 103 995.95 ($R2 = 0.600$). [85](#_Toc229829370)](#_Toc229829370)

[Ilustración 30. LLMTime-inspired — predicción sobre Electricity pl = 96 (sample 2043, seed = 2021); MSE de muestra = 131 375.16 ($R2 = 0.494$). [85](#_Toc229829371)](#_Toc229829371)

[Ilustración 31. SAX-inspired — predicción sobre Electricity pl = 96 (sample 2043, seed = 2021); MSE de muestra = 184 054.47 ($R2 = 0.292$). [85](#_Toc229829372)](#_Toc229829372)

# ÍNDICE DE TABLAS 

[Tabla 1. Comparativa formal de las seis técnicas de tokenización implementadas [21](#_Ref226879852)](#_Ref226879852)

[Tabla 2. Datasets empleados en la evaluación experimental [24](#_Ref226879964)](#_Ref226879964)

[Tabla 3. Estructura e integridad de los seis *datasets* [26](#_Ref226880013)](#_Ref226880013)

[Tabla 4. Análisis estadístico y temporal del target en cada *dataset* [27](#_Ref226880033)](#_Ref226880033)

[Tabla 5. Óptimo robusto (variante, K) por dataset, seleccionado a partir de las 192 ejecuciones del barrido de K @ $O\  = \ 96$ (mean MSE sobre 3 seeds). [31](#_Ref228520493)](#_Ref228520493)

[Tabla 6. Óptimo (variante, K) por dataset en RITMO-M (Plan B), seleccionado por argmin del MSE de validación promediado sobre los 4 horizontes (256 ejecuciones, seed = 2021). [32](#_Ref228551187)](#_Ref228551187)

[Tabla 7. Comparativa entre los dos escenarios del diseño experimental. [34](#_Ref226881947)](#_Ref226881947)

[Tabla 8. Métricas intrínsecas universales — ETTh1, mean ± std sobre 3 seeds. [40](#_Ref228534824)](#_Ref228534824)

[Tabla 9. Métricas intrínsecas universales — ETTh2, mean ± std sobre 3 seeds. [40](#_Toc229829381)](#_Toc229829381)

[Tabla 10. Métricas intrínsecas universales — Weather, mean ± std sobre 3 seeds. [41](#_Toc229829382)](#_Toc229829382)

[Tabla 11. Métricas intrínsecas universales — Electricity, mean ± std sobre 3 seeds. [41](#_Toc229829383)](#_Toc229829383)

[Tabla 12. Métricas intrínsecas discretas — ETTh1, mean ± std sobre 3 seeds. [42](#_Ref228534808)](#_Ref228534808)

[Tabla 13. Métricas intrínsecas discretas — ETTh2, mean ± std sobre 3 seeds. [42](#_Toc229829385)](#_Toc229829385)

[Tabla 14. Métricas intrínsecas discretas — Weather, mean ± std sobre 3 seeds. [42](#_Toc229829386)](#_Toc229829386)

[Tabla 15. Métricas intrínsecas discretas — Electricity, mean ± std sobre 3 seeds. [42](#_Toc229829387)](#_Toc229829387)

[Tabla 16. Robustez intrínseca — cambio relativo en MSE de reconstrucción bajo perturbación gaussiana, resumen cross-dataset. [43](#_Ref228532293)](#_Ref228532293)

[Tabla 17. Robustez intrínseca — cambio relativo en MSE de reconstrucción bajo σ = 0.5·σₓ por dataset. [43](#_Ref228532301)](#_Ref228532301)

[Tabla 18. Distancia en el espacio de tokens entre serie original y perturbada bajo σ = 0.5·σₓ, mean sobre 3 seeds. Edit distance normalizada a \[0, 1\] para técnicas discretas; L₂ en unidades del espacio de tokens para técnicas continuas. [44](#_Ref228532275)](#_Ref228532275)

[Tabla 19. Intervalos explorados, valor seleccionado y justificación de los hiperparámetros del *transformer* común. [45](#_Ref226879947)](#_Ref226879947)

[Tabla 20. Plan A — ETTh1: MSE por horizonte y MSE/MAE promedio sobre 3 seeds (mean ± std). Fila destacada en negrita: mejor avg MSE. Por columna, negrita marca el mejor valor y cursiva el segundo mejor. [47](#_Ref228534763)](#_Ref228534763)

[Tabla 21. Plan A — ETTh2: MSE por horizonte y MSE/MAE promedio sobre 3 seeds (mean ± std). Fila destacada en negrita: mejor avg MSE. Por columna, negrita marca el mejor valor y cursiva el segundo mejor. [48](#_Toc229829393)](#_Toc229829393)

[Tabla 22. Plan A — Weather: MSE por horizonte y MSE/MAE promedio sobre 3 seeds (mean ± std). Fila destacada en negrita: mejor avg MSE. Por columna, negrita marca el mejor valor y cursiva el segundo mejor. [49](#_Toc229829394)](#_Toc229829394)

[Tabla 23. Plan A — Electricity: MSE por horizonte y MSE/MAE promedio sobre 3 seeds (mean ± std). Fila destacada en negrita: mejor avg MSE. Por columna, negrita marca el mejor valor y cursiva el segundo mejor. [49](#_Toc229829395)](#_Toc229829395)

[Tabla 24. Plan A — Test pareado de Wilcoxon de signo-rango entre RITMO y los cinco baselines deterministas (n = 12 pares por celda; corrección Bonferroni × 5; α = 0.01). [51](#_Ref228534756)](#_Ref228534756)

[Tabla 25. Plan A — Traffic (transferencia cross-domain del tokenizer HMM): MSE por horizonte y MSE/MAE promedio sobre 3 seeds (mean ± std). Fila destacada en negrita: mejor avg MSE. Por columna, negrita marca el mejor valor y cursiva el segundo mejor. [53](#_Ref228535180)](#_Ref228535180)

[Tabla 26. Plan A — Exchange (transferencia cross-domain del tokenizer HMM): MSE por horizonte y MSE/MAE promedio sobre 3 seeds (mean ± std). Fila destacada en negrita: mejor avg MSE. Por columna, negrita marca el mejor valor y cursiva el segundo mejor. [54](#_Ref228535187)](#_Ref228535187)

[Tabla 27. Plan B — ETTh1: MSE por horizonte y MSE/MAE promedio (seed única = 2021, `--features M`). Por columna, negrita marca el mejor valor y cursiva el segundo mejor. [57](#_Ref228557371)](#_Ref228557371)

[Tabla 28. Plan B — ETTh2: MSE por horizonte y MSE/MAE promedio (seed única = 2021, `--features M`). Por columna, negrita marca el mejor valor y cursiva el segundo mejor. [57](#_Ref228558297)](#_Ref228558297)

[Tabla 29. Plan B — Weather: MSE por horizonte y MSE/MAE promedio (seed única = 2021, `--features M`). Por columna, negrita marca el mejor valor y cursiva el segundo mejor. [58](#_Ref228558282)](#_Ref228558282)

[Tabla 30. Plan B — Electricity: MSE por horizonte y MSE/MAE promedio (seed única = 2021, `--features M`). Por columna, negrita marca el mejor valor y cursiva el segundo mejor. [58](#_Toc229829402)](#_Toc229829402)

[Tabla 31. Comparativa de RITMO entre Plan A (`--features S`, 3 seeds) y Plan B (`--features M`, seed única 2021): configuración óptima, avg MSE/MAE y posición relativa por dataset. [61](#_Ref228557436)](#_Ref228557436)

[Tabla 32. Barrido de K — ETTh1, MSE de validación @ $O\  = \ 96$ (mean ± std sobre 3 seeds). [78](#_Toc229829404)](#_Toc229829404)

[Tabla 33. Barrido de K — ETTh2, MSE de validación @ $O\  = \ 96$ (mean ± std sobre 3 seeds). [79](#_Toc229829405)](#_Toc229829405)

[Tabla 34. Barrido de K — Weather, MSE de validación @ $O\  = \ 96$ (mean ± std sobre 3 seeds). [79](#_Toc229829406)](#_Toc229829406)

[Tabla 35. Barrido de K — Electricity, MSE de validación @ $O\  = \ 96$ (mean ± std sobre 3 seeds). [80](#_Toc229829407)](#_Toc229829407)

[Tabla 36. Barrido de K — ETTh1, Plan B: MSE de validación promedio sobre 4 horizontes (seed = 2021). [81](#_Ref228609634)](#_Ref228609634)

[Tabla 37. Barrido de K — ETTh2, Plan B: MSE de validación promedio sobre 4 horizontes (seed = 2021). [82](#_Toc229829409)](#_Toc229829409)

[Tabla 38. Barrido de K — Weather, Plan B: MSE de validación promedio sobre 4 horizontes (seed = 2021). [82](#_Toc229829410)](#_Toc229829410)

[Tabla 39. Barrido de K — Electricity, Plan B: MSE de validación promedio sobre 4 horizontes (seed = 2021). [83](#_Toc229829411)](#_Toc229829411)

# INTRODUCCIÓN

## Descripción del problema

La predicción de series temporales es un campo con aplicaciones directas en sectores como la energía, el transporte, las finanzas o la meteorología. Anticipar el consumo eléctrico, la ocupación de una red de tráfico o la evolución de un tipo de cambio permite tomar decisiones informadas con antelación, y la calidad de esas decisiones depende directamente de la precisión del modelo de predicción empleado.

Históricamente, este campo ha estado dominado por los modelos estadísticos lineales de la familia ARIMA (*AutoRegressive Integrated Moving Average*) y sus variantes (Box & Jenkins, 1976), que estiman la dependencia temporal mediante combinaciones lineales de observaciones pasadas y errores residuales. Aunque estos modelos resultaron eficaces durante décadas en series con dinámicas relativamente regulares, su capacidad para capturar relaciones no lineales y dependencias de largo plazo es limitada (De Gooijer & Hyndman, 2006), lo que motivó la búsqueda de alternativas basadas en redes neuronales capaces de aprender patrones complejos directamente desde los datos.

Las primeras arquitecturas que ofrecieron esta flexibilidad fueron las redes neuronales recurrentes (RNN), diseñadas para procesar secuencias manteniendo un estado interno que se actualiza en cada paso temporal. Su aplicación al *forecasting* reveló problemas de desvanecimiento y explosión del gradiente al modelar dependencias largas, limitación que las redes *Long Short-Term Memory* (LSTM) (Hochreiter & Schmidhuber, 1997) abordaron mediante mecanismos de puertas que controlan qué información se retiene, se olvida o se actualiza. Las LSTM —junto con su variante *Bidirectional LSTM* (BiLSTM)— se consolidaron durante años como el estándar para el modelado secuencial, pero su naturaleza recurrente impone un procesamiento estrictamente paso a paso que dificulta tanto el entrenamiento paralelo como la captura de dependencias a horizontes muy largos.

La aparición de las arquitecturas *transformer* (Vaswani et al., 2017) supuso un cambio de paradigma: al sustituir la recurrencia por un mecanismo de atención que evalúa todas las posiciones de la secuencia simultáneamente, los *transformers* permiten procesar la información de forma paralelizada y modelar dependencias de cualquier alcance sin la degradación característica de las redes recurrentes. Esta ventaja ha impulsado en los últimos años una rápida adaptación al dominio de las series temporales, dando lugar a una familia de arquitecturas especializadas: Informer (H. Zhou et al., 2021), que reduce la complejidad cuadrática de la atención mediante selección de *queries* dominantes; Autoformer (Wu et al., 2022), que integra descomposición serie–autocorrelación dentro del propio bloque de atención; PatchTST (Nie et al., 2023), que segmenta la serie en parches que actúan como *tokens*; Non-stationary Transformer (Liu et al., 2023), que mitiga el efecto de la sobre-estacionarización en la atención; y Crossformer (Y. Zhang & Yan, 2022), que captura dependencias cruzadas entre variables y entre escalas temporales. Estas arquitecturas se han consolidado como el estándar actual del *forecasting* de series temporales y constituyen el contexto técnico inmediato del presente trabajo.

Sin embargo, la transferencia de los *transformers* al dominio temporal enfrenta un problema fundamental: las series temporales son señales continuas, mientras que los *transformers* operan sobre secuencias discretas de *tokens*. Esta disparidad exige una etapa previa de *tokenización* que convierta las observaciones numéricas en representaciones procesables por el modelo, y esa etapa condiciona todo el *pipeline* posterior.

Los métodos de tokenización actuales se agrupan en cinco categorías, revisadas en detalle en el Capítulo 3. La *discretización* (*quantile-based symbolic discretization*) transforma valores continuos en símbolos finitos mediante cuantización o *codebooks* aprendidos. Los enfoques *text-based (character-serialization tokenizer)* serializan cada valor numérico como cadena de caracteres y explotan tokenizadores lingüísticos existentes. El *patching (fixed-length segment tokenization)* divide la serie en subsecuencias de longitud fija que actúan como *tokens* continuos. La *descomposición (trend–seasonal decomposition tokenization)* separa la serie en componentes aditivos —tendencia, estacionalidad, residuo— que se modelan de forma independiente. Por último, los *modelos fundacionales (foundation-inspired masked patch tokenization)* aprenden representaciones universales mediante pre-entrenamiento masivo sobre millones de series heterogéneas.

Cada una de estas categorías presenta limitaciones específicas. La discretización impone una granularidad fija que no se adapta a la variabilidad intrínseca de la serie. Los enfoques *text-based* dependen de representaciones aprendidas de corpus lingüísticos sin garantías sobre propiedades temporales. El *patching* con ventanas de longitud constante fragmenta patrones coherentes de forma arbitraria. La descomposición mantiene las dependencias entre componentes solo de forma implícita. Los modelos fundacionales requieren cantidades masivas de datos y generan *embeddings* cuyas dimensiones carecen de interpretación directa.

Lo que todas estas técnicas comparten es la ausencia de un mecanismo probabilístico explícito que modele las dependencias temporales *dentro* del propio proceso de tokenización. La estructura temporal se delega íntegramente al *transformer*, no al *token*.

## Motivación

Los *Hidden Markov Models* (HMM) ofrecen precisamente esa estructura probabilística ausente. En un HMM, una cadena de Markov de primer orden gobierna las transiciones entre estados ocultos, donde cada estado representa un régimen estadístico con parámetros de emisión propios —una media y una varianza que caracterizan las observaciones generadas en ese régimen—. Las transiciones entre estados se rigen por una matriz de probabilidad que codifica la dinámica temporal del sistema. Esta formulación permite que los estados ocultos actúen como *tokens* con significado estadístico: cada uno encapsula información sobre el régimen al que pertenece la observación y sobre la dinámica que conecta regímenes consecutivos.

A partir de un HMM entrenado es posible construir *embeddings* estructurados que concatenen las estadísticas de cada régimen —su centro, su volatilidad— con las probabilidades de transición hacia los demás estados. A diferencia de los *embeddings* implícitos de modelos fundacionales o las representaciones puramente numéricas del *patching*, cada dimensión del vector resultante tiene un significado estadístico concreto. La definición formal de estos *embeddings* se presenta en la sección 4.2.6.

Además, la teoría detrás de los HMM está bien establecida. El algoritmo de Baum-Welch (Dempster et al., 1977) garantiza convergencia monótona a un máximo local de la verosimilitud. El algoritmo de Viterbi proporciona la secuencia óptima de estados mediante programación dinámica. Y el algoritmo *Forward-Backward* permite obtener distribuciones posteriores suaves sobre los estados en cada instante temporal. No se trata de una heurística *ad hoc*, sino de un marco teórico con décadas de validación en campos como el reconocimiento de voz, la bioinformática y la modelización financiera (Hamilton, 1989; Rabiner, 1989).

Esta motivación se refuerza con un hallazgo reciente. (Y. Wang et al., 2025) demuestran que múltiples *benchmarks* de series temporales han alcanzado saturación: incrementar la complejidad del modelo apenas reduce el error, y añadir más capas o más parámetros no produce mejoras significativas. Este resultado sugiere que el progreso futuro puede venir de *cómo se representa* la serie, no de *cuánta capacidad* tiene el modelo. La tokenización basada en HMM explora exactamente esa dirección.

## Organización de la memoria

El presente documento se organiza en nueve capítulos.

En el Capítulo 1 se ha presentado el problema de la tokenización de series temporales, las limitaciones de los métodos actuales y la motivación para explorar una alternativa basada en HMM.

El Capítulo 2 define el objetivo general y los objetivos específicos del proyecto, vinculando cada objetivo con su correspondiente pregunta de investigación.

El Capítulo 3 recoge el estado del arte, con una revisión de los *transformers* aplicados a series temporales, las cinco familias de técnicas de tokenización existentes, los Hidden Markov Models como marco probabilístico complementario, los avances en normalización y preprocesamiento, el fenómeno de saturación de *benchmarks* y, finalmente, el *gap* identificado en la literatura que motiva la propuesta.

El Capítulo 4 describe la metodología: el *pipeline* completo del sistema RITMO, el marco teórico que formaliza los algoritmos y conceptos matemáticos empleados, las herramientas y tecnologías utilizadas, los *datasets* seleccionados y las métricas de evaluación.

El Capítulo 5 presenta el desarrollo técnico, comenzando con el diseño experimental que define los dos escenarios complementarios (Plan A y Plan B) y continuando con la implementación y validación de cada fase del *pipeline*, junto con la comparativa intrínseca entre las seis técnicas de tokenización.

El Capítulo 6 reporta los resultados experimentales: el Plan A —tanto *in-domain* como en transferencia *cross-domain* del *tokenizer* HMM congelado—, la inspección cualitativa de los regímenes aprendidos, y el Plan B con la comparación de RITMO-M frente a los *baselines* DLinear, PatchTST, TimeMixer y TimeXer.

El Capítulo 7 discute estos resultados respondiendo a tres preguntas: qué demuestran, qué no demuestran y dónde funciona o no funciona RITMO.

El Capítulo 8 sintetiza las conclusiones generales, las limitaciones del estudio y las líneas futuras de trabajo.

El Capítulo 9 recoge las referencias bibliográficas citadas a lo largo del documento.

# OBJETIVOS

## Objetivo general

Las técnicas existentes de tokenización de series temporales —*discretización*, *text-based*, *patching*, *descomposición* y *modelos fundacionales*— han demostrado utilidad en distintos *benchmarks*, pero todas comparten una limitación estructural común: ninguna incorpora un mecanismo probabilístico explícito que modele las dependencias temporales dentro del propio proceso de tokenización. La estructura temporal queda delegada íntegramente al *transformer*, lo que reduce la representación de entrada a una transformación pasiva sin contenido dinámico propio. Esta carencia, esbozada en el Capítulo 1 y profundizada en el Capítulo 3, motiva explorar alternativas que doten a la propia tokenización de estructura probabilística.

Los Hidden Markov Models (HMM) ofrecen exactamente esa estructura ausente. Una matriz de transición codifica explícitamente las dinámicas entre regímenes, los parámetros de emisión cuantifican el centro y la volatilidad de cada estado, y un marco teórico consolidado garantiza inferencia, decodificación y entrenamiento bien definidos (Dempster et al., 1977; Rabiner, 1989). Reinterpretar los estados ocultos de un HMM como *tokens* —y construir *embeddings* estructurados a partir de sus parámetros estadísticos— abre la posibilidad de una tokenización con estructura probabilística explícita e interpretabilidad directa, propiedades simultáneamente ausentes en las cinco técnicas determinísticas anteriores.

Por tanto, el objetivo general planteado en este trabajo es **proponer, implementar y evaluar RITMO (Regímenes latentes mediante Inferencia Temporal con Markov Oculto), un sistema de tokenización de series temporales basado en estados ocultos de Hidden Markov Models que actúan como *embeddings* latentes estructurados con significado estadístico explícito**. La validación combina (i) una comparación controlada (Plan A) frente a las cinco técnicas determinísticas existentes —discretización, *text-based*, *patching*, descomposición y *modelos fundacionales*— bajo condiciones experimentales idénticas, y (ii) una comparación con el estado del arte (Plan B) frente a cuatro *baselines* consolidados —DLinear, PatchTST, TimeMixer y TimeXer—, con el fin de cuantificar tanto la contribución específica de la tokenización probabilística como su competitividad global en predicción a largo plazo.

## Objetivos específicos

Cada objetivo específico responde a una pregunta de investigación (RQ) planteada en el anteproyecto del proyecto.

**OE1 (RQ1).** Implementar las seis técnicas de tokenización con sus *embeddings* naturales correspondientes. Para garantizar comparabilidad, todas las técnicas comparten un *backbone transformer* único y producen representaciones en un espacio común de dimensión $\left\lbrack seq,d_{model} \right\rbrack$.

**OE2 (RQ2).** Implementar el *pipeline* RITMO completo —normalización RevIN, entrenamiento Baum-Welch, tokenización Viterbi y generación de *embeddings* estructurados— y validar cada fase de forma independiente bajo el protocolo multi-seed (3 *seeds* {42, 2021, 7}): error de reversibilidad en la normalización, convergencia monótona de la log-verosimilitud en el entrenamiento, optimalidad de la secuencia decodificada y separación de regímenes en el espacio de *embeddings*.

**OE3 (RQ3).** Establecer un protocolo de evaluación dual que combine métricas intrínsecas de tokenización —ratio de compresión, error de reconstrucción, retención de autocorrelación, entropía de vocabulario, persistencia de *tokens* y robustez bajo perturbación— con métricas de predicción *downstream* —MSE y MAE— sobre *benchmarks* consolidados, agregadas sobre 3 *seeds* y contrastadas mediante test pareado de Wilcoxon (Bonferroni × 5, α = 0.01).

**OE4 (RQ4).** Ejecutar una comparación controlada de las seis técnicas (Plan A) con un mismo *transformer*, evaluando los *trade-offs* entre compresión, preservación de información y capacidad predictiva en los horizontes {96, 192, 336, 720} sobre cuatro *datasets* de entrenamiento y dos *datasets* en modo de transferencia *cross-domain* del *tokenizer* HMM congelado (referido como evaluación *zero-shot* en el sentido restringido descrito en la sección 5.1.1), con todas las cifras agregadas sobre 3 *seeds* {42, 2021, 7}.

**OE5 (RQ5).** Ejecutar una comparación con el estado del arte (Plan B) enfrentando RITMO-M —el HMM 1D del Plan A reentrenado en modo --features M con *channel-independence*, para garantizar paridad con el régimen multivariado en que los *baselines* publican sus resultados— contra DLinear, PatchTST, TimeMixer y TimeXer en sus configuraciones óptimas publicadas, sobre los cuatro *datasets* del Grupo 1 con seed única (2021) por restricción de presupuesto GPU, cuantificando las ganancias o pérdidas atribuibles a la tokenización probabilística frente a arquitecturas especializadas que integran su propia representación.

# ESTADO DEL ARTE

Este capítulo recoge la literatura relevante para el presente trabajo, organizada según un hilo conductor que parte del contexto más amplio —la consolidación de los *transformers* como herramienta de modelado de secuencias y su adaptación al dominio temporal— y desciende progresivamente hacia el problema específico que motiva la propuesta: la ausencia de mecanismos probabilísticos en la tokenización de series temporales.

La revisión se estructura en siete bloques temáticos. El primero presenta los *surveys* recientes y la conexión entre Large Language Models (LLMs) y series temporales, contextualizando el papel de los *transformers* como arquitectura de referencia. El segundo recorre las cinco familias de técnicas de tokenización existentes con sus aportaciones y limitaciones. El tercero amplía el análisis con los modelos *transformer* especializados que constituyen los *baselines* del Plan B. El cuarto introduce los Hidden Markov Models como marco probabilístico complementario que ofrece la estructura temporal ausente en las técnicas anteriores. El quinto revisa los avances en normalización y preprocesamiento que mitigan el *distribution shift*, factor crítico en series no estacionarias. El sexto documenta el fenómeno de saturación de *benchmarks*, que motiva buscar el progreso en la representación más que en la capacidad del modelo. Finalmente, el séptimo bloque sintetiza el *gap* identificado en la literatura, que constituye el punto de partida de RITMO.

## LLMs y transformers en series temporales

La aplicación de arquitecturas *transformer* al dominio temporal ha experimentado un desarrollo acelerado en los últimos años, y varios *surveys* recientes sistematizan este campo desde perspectivas complementarias. Abdullahi et al. (2025) identifican tres paradigmas principales: la adaptación de Large Language Models (LLMs) pre-entrenados mediante *fine-tuning*, las arquitecturas diseñadas específicamente para series temporales, y los métodos de tokenización que transforman observaciones continuas en secuencias discretas. En esa misma línea, Zhang et al. (2024) documentan limitaciones concretas en la preservación de dependencias para horizontes largos, mientras que Jiang et al. (2024) señalan carencias en la captura de dependencias complejas mediante atención estándar, lo que motiva la búsqueda de alternativas con estructura explícita. Desde una óptica más amplia, Liang et al. (2024) proponen una taxonomía que clasifica *foundation models* según arquitectura, técnicas de pre-entrenamiento y estrategias de adaptación, proporcionando un marco dentro del cual los estados ocultos de un HMM encajan como *embeddings* con estructura probabilística. Wen et al. (2023) ofrecen una panorámica de las arquitecturas *transformer* consolidadas que constituyen el contexto arquitectónico del presente trabajo.

La conclusión común de estos *surveys* es que la tokenización sigue siendo un desafío abierto: convertir señales continuas en representaciones discretas eficaces condiciona el rendimiento de todo el *pipeline* posterior.

## Técnicas de tokenización

La literatura actual recoge cinco familias de técnicas de tokenización para series temporales. A continuación se revisa cada una con sus principales aportaciones y limitaciones.

**Discretización (*quantile-based symbolic discretization*).** La representación simbólica de series temporales tiene su origen en SAX (*Symbolic Aggregate approXimation*), propuesta por Lin et al. (2007), que reduce la dimensionalidad mediante agregación y discretización alfabética con garantía de *lower-bound* sobre la distancia euclídea. Su principal limitación es la granularidad fija y la ausencia de memoria temporal. Para superar esta rigidez, enfoques posteriores han introducido *codebooks* aprendidos: Oord et al. (2018) proponen VQ-VAE con entrenamiento *end-to-end*, y Talukder et al. (2025) lo adaptan al dominio temporal en TOTEM, generando un vocabulario de aproximadamente 256 *tokens* con ratio de compresión 4:1. Más recientemente, Ansari et al. (2024) desarrollan Chronos, que combina *mean scaling* con cuantización uniforme en un vocabulario de aproximadamente 4096 *tokens* y entrena arquitecturas T5/GPT-2 sin modificaciones, incorporando *data augmentation* mediante TSMixup y KernelSynth. Zhao et al. (2024) completan esta línea con Sparse-VQ, que aplica *vector quantization* tras el *encoder* para discretizar *embeddings* y reducir ruido. A pesar de estos avances, todas las variantes de discretización comparten una limitación fundamental: los símbolos se asignan de forma puntual, sin incorporar información sobre las transiciones temporales entre estados consecutivos.

**Patching (*fixed-length segment tokenization*).** Una alternativa influyente es la segmentación de la serie en subsecuencias contiguas que actúan como *tokens* continuos. PatchTST (Nie et al., 2023) introduce este enfoque y reduce la complejidad de atención de $O\left( L^{2} \right)\text{ a }O\left( \left( \frac{L}{S} \right)^{2} \right)$ mediante *channel-independence* con pesos compartidos. Trabajos posteriores refinan la estrategia de segmentación: Abeywickrama et al. (2026) proponen EntroPE, donde los límites de cada *patch* se determinan mediante entropía condicional en puntos de alta incertidumbre que marcan transiciones naturales de la serie, y Peršak et al. (2025) exploran *patching* multi-resolución dividiendo la serie a múltiples escalas simultáneamente. En el ámbito multivariado, Y. Zhang & Yan (2022) combinan la segmentación con atención *Two-Stage* en Crossformer para modelar dependencias tanto temporales como entre variables. La limitación compartida del *patching* es que la longitud de ventana —fija o adaptativa— determina qué patrones se capturan y cuáles se fragmentan, sin que exista un criterio probabilístico que guíe la segmentación.

**Descomposición (*trend–seasonal decomposition tokenization*).** Un tercer paradigma separa la serie en componentes aditivos que se modelan de forma independiente. Autoformer (Wu et al., 2022) lo aborda mediante módulos de autocorrelación progresiva, mientras que FEDformer (T. Zhou et al., 2022) opera en el dominio frecuencial aplicando la transformada de Fourier para capturar periodicidades de forma eficiente. DLinear (Zeng et al., 2023) demuestra que un enfoque más simple —descomponer mediante promedio móvil y aplicar dos capas lineales separadas a tendencia y estacionalidad— puede superar a *transformers* complejos en múltiples *benchmarks*, resultado que cuestiona la necesidad de arquitecturas sofisticadas cuando la representación es adecuada. Cao et al. (2024) adaptan GPT-2 al dominio temporal en TEMPO mediante descomposición STL con *prompts semi-soft* por componente, y Woo et al. (2022) aprenden representaciones desacopladas *seasonal-trend* mediante *contrastive learning* en CoST. Aunque la descomposición facilita el aprendizaje al separar dinámicas de distinta escala, las dependencias entre componentes solo se preservan de forma implícita y no se modelan transiciones entre regímenes.

**Foundation models (*foundation-inspired masked patch tokenization*).** La cuarta familia busca representaciones universales mediante pre-entrenamiento masivo combinado con tokenización por *patches* enmascarados. MOMENT (Goswami et al., 2024) se pre-entrena sobre *Time Series Pile* con *masked reconstruction* de un 30% de *patches* enmascarados. Timer (Liu et al., 2024) adopta un enfoque *decoder-only* estilo GPT entrenado sobre datos de múltiples dominios. MOIRAI (Woo et al., 2024) introduce un mecanismo de *any-variate attention* que unifica series univariadas y multivariadas con soporte multi-frecuencia. En una dirección complementaria, FPT (T. Zhou et al., 2023) emplea un LLM *frozen* como *encoder* de secuencias temporales previamente parcheadas y proyectadas linealmente. Estos modelos alcanzan resultados competitivos, pero sus *embeddings* de alta dimensionalidad carecen de interpretación directa: las dimensiones del vector latente no corresponden a magnitudes observables ni a propiedades estadísticas de la serie.

**Text-based (*character-serialization tokenization*).** La quinta familia explota directamente los *tokenizers* lingüísticos de los LLMs mediante la serialización carácter a carácter de los valores numéricos. LLMTime (Gruver et al., 2024) convierte series temporales en cadenas numéricas procesadas por GPT-3/LLaMA-2 mediante *next-token prediction*, alcanzando competitividad con métodos especializados en condiciones de transferencia sin re-entrenamiento del modelo de lenguaje. Time-LLM (Jin et al., 2024) extiende esta idea reprogramando LLMs mediante *text prototypes* aprendidos que alinean *embeddings* temporales con el espacio lingüístico. La dependencia de *tokenizers* diseñados para lenguaje natural implica que la representación numérica hereda sesgos del corpus de texto original, sin garantías de que las propiedades temporales se preserven.

## Transformers especializados para series temporales

Más allá de las técnicas de tokenización propiamente dichas, la literatura recoge una familia de arquitecturas *transformer* desarrolladas específicamente para *forecasting* de series temporales que constituyen los *baselines* contra los que se evalúa este trabajo. A diferencia de las técnicas anteriores —centradas en cómo representar la entrada—, los modelos de esta sección integran su propia tokenización dentro de arquitecturas optimizadas para la tarea predictiva, lo que permite contrastar el efecto de la representación frente al de la arquitectura completa.

Informer (H. Zhou et al., 2021) introduce *ProbSparse self-attention*, que reduce la complejidad a $O\ \left( L\  \bullet \log L \right)$ mediante selección de *queries* dominantes. TimesNet (Wu et al., 2023) transforma series 1D en tensores 2D y aplica convoluciones para capturar variaciones intra e inter-período. TimeMixer (S. Wang et al., 2024) implementa descomposición *multiscale mixing* con operaciones separadas por escala temporal. TimeXer (Y. Wang et al., 2024) incorpora variables exógenas mediante *exogenous-aware attention* que modula la atención endógena con información externa.

Estos modelos comparten una característica relevante para el presente trabajo: la estructura temporal se aprende exclusivamente mediante el mecanismo de atención, sin que la tokenización aporte información temporal por sí misma. La representación de entrada es agnóstica respecto a las dependencias entre observaciones consecutivas.

## Hidden Markov Models

La limitación recién identificada —tokenización sin contenido temporal explícito— motiva explorar marcos probabilísticos en los que la propia representación incorpore estructura dinámica. Los Hidden Markov Models constituyen el ejemplo más consolidado de este tipo de modelos, con décadas de validación en dominios donde las secuencias se generan a partir de estados latentes con dinámica markoviana, desde el reconocimiento de voz hasta la modelización financiera.

Los HMM tienen una trayectoria consolidada en la modelización de secuencias con estados latentes. Dempster et al. (1977) formalizan el algoritmo EM (*Expectation-Maximization*) para estimación de máxima verosimilitud con datos incompletos, demostrando convergencia monótona, con aplicación directa a HMM. Rabiner (1989) sistematiza la teoría completa mediante un tutorial que presenta los tres algoritmos fundamentales: *Forward-Backward* para evaluación, Viterbi para decodificación y Baum-Welch para entrenamiento de parámetros.

En el ámbito de *regime-switching*, Hamilton (1989) introduce el modelo de *Markov-Switching*, en el que los parámetros de autoregresión cambian según un estado latente discreto, y demuestra que el ciclo económico de EE.UU. se caracteriza mejor mediante cambios discretos entre regímenes que mediante modelos ARIMA lineales. Más recientemente, Tang & Matteson (2021) combinan *State-Space Models* con *transformers* en ProTran, validando que estados ocultos con dinámica temporal explícita funcionan como *embeddings* intermedios eficaces para *forecasting* probabilístico.

Las extensiones modernas del marco HMM son también relevantes para este trabajo. Fox et al. (2011) proponen *sticky HDP-HMM*, que añade un parámetro de auto-transición para controlar la persistencia temporal y evitar sobre-segmentación. Dai et al. (2017) extienden el HSMM (*Hidden Semi-Markov Model*) clásico reemplazando emisiones paramétricas por RNNs generativas por estado. Mensch & Blondel (2018) hacen diferenciables los algoritmos de programación dinámica —incluido Viterbi— mediante suavizado del operador *max*, lo que permite *backpropagation* a través de la decodificación. Yeh & Tang (2022) desarrollan *Neural HMMs* con dependencias markovianas explícitas, reportando ganancias significativas frente a métodos sin estructura temporal en tareas de segmentación. Estas extensiones confirman que el marco HMM sigue siendo fértil y compatible con las arquitecturas modernas de *deep learning*.

## Normalización y preprocesamiento

Junto a la elección de la tokenización, el éxito de cualquier sistema de *forecasting* depende del tratamiento previo de la serie. Las series temporales reales presentan habitualmente cambios estadísticos entre las particiones de entrenamiento y test —el llamado *distribution shift*—, lo que degrada el desempeño de cualquier modelo, incluido un HMM o un *transformer*, si no se aborda de forma explícita en el preprocesamiento.

Kim et al. (2021) abordan este problema con RevIN (*Reversible Instance Normalization*), que normaliza cada instancia almacenando las estadísticas de media y desviación para desnormalización posterior. Liu et al. (2023) identifican un efecto adverso de la normalización excesiva —la sobre-estacionarización, que genera atenciones indistinguibles— y proponen *Non-stationary Transformers* con *De-stationary Attention* para mitigarlo. Jibao et al. (2025) refinan el enfoque mediante *Inner-instance Normalization* con ventanas deslizantes adaptativas que capturan estadísticas locales, demostrando mayor robustez frente a no-estacionariedad extrema.

## Saturación de benchmarks

A pesar de los avances en arquitecturas, técnicas de tokenización y estrategias de normalización, un estudio reciente sugiere que el progreso en *benchmarks* consolidados de *forecasting* se ha estancado. Este hallazgo redefine el horizonte de mejora y orienta la búsqueda de avances hacia la representación de la serie más que hacia la capacidad del modelo.

La comunidad ha consolidado *benchmarks* estandarizados mediante TSLib (*Time-Series-Library*), que proporciona implementaciones unificadas y protocolos experimentales consistentes. El protocolo estándar, establecido por Informer (H. Zhou et al., 2021), emplea una ventana de entrada $I = 96$ *timesteps* y horizontes de predicción $O\  \in \text{\{}96,\ 192,\ 336,\ 720\text{\}}$, evaluando mediante MSE (*Mean Squared Error*) y MAE (*Mean Absolute Error*).

En este contexto, Y. Wang et al. (2025) establecen un hallazgo que redefine expectativas: una *Accuracy Law* que formaliza la relación $MSE \approx \exp\left( \alpha \cdot \text{Complexity} \right) - 1$, donde $\alpha$ es una constante del *dataset* y *Complexity* cuantifica la dificultad intrínseca de la tarea. La evidencia empírica demuestra saturación en múltiples *benchmarks* —ETTh1, ETTh2, Weather, Electricity—, donde los métodos determinísticos se aproximan asintóticamente a un límite teórico. Este resultado implica que el progreso futuro no vendrá de modelos más grandes, sino de representaciones cualitativamente distintas — exactamente la dirección que explora este trabajo.

## Gap identificado

A pesar de los avances en tokenización y de la efectividad demostrada de los HMM en modelización temporal, no existe en la literatura una propuesta que emplee estados ocultos de un HMM como mecanismo de tokenización para *transformers* aplicados a series temporales. Las técnicas actuales presentan carencias complementarias: discretización y *patching* carecen de estructura probabilística; *text-based* y *foundation models* producen *embeddings* sin interpretabilidad directa; y la descomposición no modela transiciones entre regímenes. Los HMM ofrecen simultáneamente estructura probabilística rigurosa, *embeddings* interpretables con parámetros estadísticos explícitos y captura de dinámica temporal mediante la matriz de transición. Esta combinación, no explorada hasta la fecha, motiva el presente trabajo.

# METODOLOGÍA

Este capítulo describe la metodología empleada para dar respuesta a los objetivos planteados en el capítulo anterior. Se presenta el *pipeline* completo del sistema RITMO, el marco teórico que formaliza los algoritmos y conceptos matemáticos en que se apoya, las herramientas y tecnologías utilizadas, los *datasets* seleccionados, las métricas de evaluación y, finalmente, el protocolo multi-seed y la estrategia de selección del número de regímenes K que rigen todos los experimentos.

## Descripción del marco metodológico utilizado

El sistema RITMO sigue un *pipeline* de cinco fases secuenciales que transforma una serie temporal cruda en predicciones a largo plazo.

La primera fase aplica normalización reversible RevIN (Kim et al., 2021) para eliminar diferencias de escala entre instancias, almacenando las estadísticas originales —media y desviación— para restaurar la escala en la salida. La segunda fase entrena un HMM con K estados y emisiones gaussianas sobre la serie normalizada mediante el algoritmo de Baum-Welch (sección 4.2.3), estimando los parámetros óptimos $\lambda = (A,B,\pi)$. La tercera fase aplica el algoritmo de Viterbi (sección 4.2.4) —o, alternativamente, el algoritmo *Forward-Backward* (sección 4.2.2)— para asignar cada *timestep* a un estado oculto, generando una secuencia de *tokens* discretos o una distribución suave sobre los estados. La cuarta fase construye los *embeddings* estructurados definidos formalmente en la sección 4.2.6, donde cada estado se representa como un vector que concatena sus estadísticas de emisión con sus probabilidades de transición. La quinta fase alimenta estos *embeddings* a un *transformer* que produce las predicciones, cuya escala se restaura mediante la desnormalización RevIN.

Este *pipeline* admite dos modos de operación según el plan experimental. En el **Plan A** (sección 5.1.1) el HMM se entrena en modo univariante (--features S) sobre la columna objetivo OT y el *transformer* recibe y predice una única serie. En el **Plan B** (sección 5.1.2) el mismo HMM 1D se entrena en modo *channel-independent* (--features M) sobre los C canales del *dataset* apilados como C secuencias independientes —siguiendo la convención de TSLib y PatchTST—; los parámetros estimados (μₖ, σₖ, A, π) son escalares y compartidos entre canales, y el *transformer* procesa los C canales simultáneamente. Esta variante se identifica como **RITMO-M** y permite la comparación equitativa con los *baselines* del estado del arte, que publican sus resultados en este mismo modo.

La Ilustración 1 esquematiza las cinco etapas del *pipeline* y las conexiones entre ellas.

<span id="_Ref226788084" class="anchor"></span>Ilustración 1. Pipeline RITMO: cinco etapas del flujo para predicción de series temporales con HMM

<img src="./media/image3.png" style="width:6.10208in;height:2.58667in" alt="Diagrama El contenido generado por IA puede ser incorrecto." />Fuente: Elaboración propia

Este diseño separa explícitamente la **representación** temporal —responsabilidad del HMM— del **modelado** predictivo —responsabilidad del *transformer*—. La separación permite evaluar la contribución de la tokenización de forma aislada: si se sustituye la tokenización HMM por cualquiera de las cinco técnicas alternativas manteniendo el mismo *transformer*, las diferencias en el desempeño son atribuibles exclusivamente a la representación.

Adicionalmente, todas las fases con componentes estocásticos —inicialización *k-means* del HMM e inicialización Xavier del *transformer*— se ejecutan en el Plan A bajo el protocolo multi-seed (3 *seeds* {42, 2021, 7}) descrito en la sección 4.7, de modo que las cifras del Plan A reportadas en los Capítulos 5 y 6 son agregados mean ± std sobre dichas *seeds* y no resultados de un único punto. El Plan B se ejecuta con seed única (2021) por restricción de presupuesto GPU; sus cifras se reportan como valores puntuales.

## Marco Teórico

Antes de detallar las herramientas, *datasets* y diseño experimental empleados, conviene fijar los fundamentos teóricos sobre los que se apoya el *pipeline* RITMO descrito en la sección 4.1. Esta sección formaliza los algoritmos y conceptos matemáticos que sustentan cada una de las cinco fases del *pipeline*: la definición del HMM y sus parámetros (4.2.1), los algoritmos de inferencia *Forward-Backward* (4.2.2) y Viterbi (4.2.4), el procedimiento de entrenamiento Baum-Welch (4.2.3), la normalización reversible RevIN (4.2.5) y la construcción de los *embeddings* estructurados que constituyen la propuesta central del trabajo (4.2.6). El capítulo cierra con una formalización comparada de las seis técnicas de tokenización implementadas (4.2.7), donde el HMM se sitúa frente a las cinco alternativas determinísticas revisadas en el Capítulo 3 —*discretización*, *text-based*, *patching*, *descomposición* y *foundation models*— y con la descripción del *backbone* *Transformer* común (4.2.8) que recibe los *embeddings* de cada técnica. Cada uno de estos componentes se desarrolla a continuación con sus ecuaciones, propiedades fundamentales y referencias bibliográficas; su implementación práctica y los detalles de validación se documentan en el Capítulo 5.

## Hidden Markov Models

Un HMM se define mediante el conjunto de parámetros $\lambda = (A,B,\pi)$. La matriz de transición A, de dimensión $K\  \times K$, recoge las probabilidades de transición entre estados ocultos, con $A_{ij} = P\left( q_{t} = j\mid q_{t - 1} = i \right)$. Las distribuciones de emisión B especifican la probabilidad de observar cada valor dado un estado; en este trabajo se emplean emisiones gaussianas con parámetros $\left( \mu_{k},\sigma_{k} \right)$ para cada estado $k$, de modo que $b_{k}(o)\mathcal{= N}\left( o;\,\mu_{k},\,\sigma_{k}^{2} \right)$. La distribución inicial $\pi$ define la probabilidad de comenzar en cada estado, con $\pi_{k} = P\left( q_{1} = k \right)$.

La probabilidad de una secuencia de observaciones $O = \left( o_{1},\ldots,o_{T} \right)$ se obtiene marginalizando sobre todas las posibles secuencias de estados ocultos:

$$P\left( O \middle| \lambda \right) = \sum_{Q}^{}{P\left( O \middle| Q,\lambda \right)} \cdot P\left( Q \middle| \lambda \right)$$

La propiedad markoviana —el estado actual depende únicamente del estado previo— permite resolver esta suma mediante programación dinámica en lugar de enumeración exhaustiva.

Esta formulación es univariada por construcción $\left( o_{t} \in \mathbb{R} \right)$. En el Plan B (sección 5.1.2) se extiende a *channel-independence* sin alterar el formalismo: los C canales del *dataset* se apilan como C secuencias independientes y se estima un único conjunto de parámetros λ compartido entre todos ellos mediante la versión por lotes del algoritmo (sección 4.2.3).

## Algoritmo Forward-Backward

El algoritmo *Forward-Backward* calcula las probabilidades marginales $P\left( q_{t} = k\mid O,\lambda \right)$ mediante dos pasadas sobre la secuencia.

La variable *forward* $\alpha_{t}(i) = P\left( o_{1},\ldots,o_{t},\, q_{t} = i\mid\lambda \right)$ se calcula recursivamente:

$$\alpha_{1}(i) = \pi_{i} \cdot b_{i}\left( o_{1} \right)$$

$$\alpha_{t}(j) = \left\lbrack \sum_{i}^{}{\alpha_{t - 1}(i)} \cdot a_{ij} \right\rbrack \cdot b_{j}\left( o_{t} \right)$$

La variable *backward* $\beta_{t}(i) = P\left( o_{t + 1},\ldots,o_{T}\mid q_{t} = i,\lambda \right)$ se calcula en dirección inversa:

$$\beta_{T}(i) = 1$$

$$\beta_{t}(i) = \sum_{j}^{}a_{ij} \cdot b_{j}\left( o_{t + 1} \right) \cdot \beta_{t + 1}(j)$$

Combinando ambas se obtiene la probabilidad *a posteriori* de cada estado en cada instante:

$$\gamma_{t}(i) = \frac{\alpha_{t}(i) \cdot \beta_{t}(i)}{P\left( O \middle| \lambda \right)}$$

Y la probabilidad conjunta de transición entre pares de estados:

$$\xi_{t}(i,j) = \frac{\alpha_{t}(i) \cdot a_{ij} \cdot b_{j}\left( o_{t + 1} \right) \cdot \beta_{t + 1}(j)}{P\left( O \middle| \lambda \right)}$$

La complejidad es $O\left( T \cdot K^{2} \right)$, lineal en la longitud de la secuencia y cuadrática en el número de estados. En la implementación, todos los cálculos se realizan en espacio logarítmico para evitar *underflow* numérico con secuencias largas (Rabiner, 1989).

## Algoritmo de Baum-Welch

El algoritmo de Baum-Welch estima los parámetros $\lambda^{*} = argmax_{\lambda}P\left( O \middle| \lambda \right)$ mediante *Expectation-Maximization* (Dempster et al., 1977), alternando entre dos pasos hasta convergencia.

En el paso E (*Expectation*) se calculan $\gamma_{t}(i)$ y $\xi_{t}(i,j)$ usando *Forward-Backward* con los parámetros actuales. En el paso M (*Maximization*) se actualizan los parámetros usando las esperanzas calculadas:

$$\pi_{i} = \gamma_{1}(i)$$

$$a_{ij} = \frac{\sum_{t}^{}{\xi_{t}(i,j)}}{\sum_{t}^{}{\gamma_{t}(i)}}$$

$$\mu_{i} = \frac{\sum_{t}^{}{\gamma_{t}(i)} \cdot o_{t}}{\sum_{t}^{}{\gamma_{t}(i)}}$$

$$\sigma_{i}^{2} = \frac{\sum_{t}^{}{\gamma_{t}(i)} \cdot \left( o_{t} - \mu_{i} \right)^{2}}{\sum_{t}^{}{\gamma_{t}(i)}}$$

El algoritmo EM garantiza que la log-verosimilitud no decrece en cada iteración. La convergencia se determina cuando el incremento absoluto $|\Delta LL|$ cae por debajo de un umbral $\varepsilon$. La inicialización se realiza mediante *K-means* sobre las observaciones, lo que proporciona centros iniciales razonables y reduce el riesgo de convergencia a máximos locales pobres.

Para el Plan B se utiliza una **versión por lotes** del algoritmo que recibe un tensor de B secuencias de longitud T y agrega las estadísticas suficientes $\sum_{b}^{}{\gamma_{t}^{(b)}(i)}$ y $\sum_{b}^{}{\xi_{t}^{(b)}(i,j)}$ a través de las B secuencias en cada iteración EM, manteniendo un único conjunto de parámetros λ compartido entre todas. Apilando los C canales del *dataset* como B = C secuencias se obtiene el HMM *channel-independent* que tokeniza el modo --features M (sección 4.2.1).

## Algoritmo de Viterbi

El algoritmo de Viterbi encuentra la secuencia de estados más probable $Q^{*} = argmax_{Q}P\left( Q \middle| O,\ \lambda \right)\ $mediante programación dinámica:

$$\delta_{1}(i) = \ \pi_{i} \cdot b_{i}\left( o_{1} \right)$$

$$\delta_{t}(j) = max_{i}\left\lbrack \delta_{\left\{ t - 1 \right\}}(i) \cdot a_{\left\{ ij \right\}} \right\rbrack \cdot b_{j}\left( o_{t} \right)$$

$$\psi_{t}(j) = \ argmax_{i}\left\lbrack \delta_{\left\{ t - 1 \right\}}(i) \cdot a_{\left\{ ij \right\}} \right\rbrack$$

La secuencia óptima se recupera por *backtracking* desde el estado final $q_{T} = {argmax}_{i}\delta_{T}(i)$*, recorriendo los punteros ψ almacenados. La complejidad es* $O\left( T \cdot K^{2} \right)$*, idéntica a la de Forward-Backward*.

En el contexto de este trabajo, Viterbi asigna cada *timestep* a un estado latente, generando la secuencia de tokens$\left\lbrack z_{1},z_{2},\ldots,z_{T} \right\rbrack\quad\text{con}\quad z_{t} \in \text{\{}1,\ldots,K\text{\}}$. Esta secuencia constituye la tokenización de la serie temporal.

## Reversible Instance Normalization

RevIN (Kim et al., 2021) normaliza cada instancia de forma independiente antes de procesarla y desnormaliza la predicción para devolverla a la escala original:

$$X_{norm} = \frac{X - \mu}{\sigma}$$

$$ŷ = \ ŷ_{norm} \bullet \ \sigma + \ \mu$$

donde $\mu$ y $\sigma$ son la media y desviación estándar de la instancia de entrada. Este procedimiento mitiga el *distribution shift* entre entrenamiento y test sin añadir parámetros al modelo. En el *pipeline* RITMO, RevIN se aplica antes del entrenamiento HMM y la desnormalización se aplica tras la predicción del *transformer*.

## Embeddings estructurados

La propuesta central de este trabajo consiste en reinterpretar los estados ocultos del HMM como *embeddings* vectoriales. A partir de los parámetros entrenados, cada estado k se representa como:

$$e_{k} = \left\lbrack \mu_{k},\sigma_{k},A\lbrack k,:\rbrack \right\rbrack \in \mathbb{R}^{(\mathbb{2} + K)}$$

donde $\mu_{k}\ $captura el centro del régimen, $\sigma_{k}\ $cuantifica su volatilidad y $A\lbrack k,:\rbrack\ $codifica las probabilidades de transición hacia los demás estados. Cada dimensión del vector tiene un significado estadístico concreto, a diferencia de los *embeddings* latentes de *foundation models* donde las dimensiones no admiten interpretación directa.

Para alimentar un *transformer* con dimensión $d_{model}$, se aplica una proyección lineal:

$$e_{k}^{'} = W \cdot e_{k} + b,\quad\text{con}\quad W \in \mathbb{R}^{(d_{model} \times \left( \mathbb{2} + K \right))}$$

Esta proyección es entrenable *end-to-end* junto con el *transformer*, permitiendo que el modelo ajuste la representación durante el aprendizaje de la tarea de predicción.

## Técnicas de tokenización comparadas

Para situar la tokenización HMM dentro del panorama más amplio de los métodos de pretratamiento revisados en el Capítulo 3, este trabajo implementa también las cinco familias de técnicas determinísticas alternativas —*discretización*, *text-based*, *patching*, *descomposición* y *foundation models*—. Cada una recibe como entrada la misma serie normalizada mediante RevIN (sección 4.2.5) y produce una secuencia de *embeddings* en un espacio común de dimensión $\left\lbrack seq,d_{model} \right\rbrack,\ $alimentada al mismo *backbone transformer* en los experimentos del Plan A descritos en la sección 5.1.1. Esta uniformidad de condiciones permite aislar el efecto de la tokenización sobre el desempeño predictivo. La Ilustración 2 muestra la serie ETTh2 normalizada que sirve de entrada común a todas las técnicas a lo largo de los ejemplos siguientes.

<span id="_Ref226879634" class="anchor"></span>Ilustración 2. Serie ETTh2 normalizada con RevIN, base común para los ejemplos de las seis técnicas de tokenización.

<img src="./media/image4.png" style="width:6.29861in;height:1.23264in" />Fuente: Elaboración propia.

**Discretización (***quantile-based symbolic discretization***, SAX-inspired).** Siguiendo la formulación de Lin et al. (2007), cada observación normalizada se discretiza en un número fijo de símbolos mediante *breakpoints* gaussianos equidistantes. El *embedding* se obtiene mediante una tabla de consulta aprendible, donde cada símbolo se mapea a un vector denso de dimensión $d_{model}$. La granularidad es fija —un *token* por observación— y no incorpora memoria temporal: dos observaciones consecutivas con el mismo símbolo reciben *embeddings* idénticos independientemente de su contexto. La Ilustración 3 ilustra los *breakpoints* gaussianos y la asignación resultante de símbolos sobre la serie ETTh2.

<span id="_Ref226879647" class="anchor"></span>Ilustración 3. Discretización (SAX-inspired) sobre ETTh2: breakpoints gaussianos y asignación de símbolos discretos.

<img src="./media/image5.png" style="width:6.29861in;height:3.7375in" />Fuente: Elaboración propia.

**Text-based (***character-serialization tokenizer***, LLMTime-inspired).** Inspirada en Gruver et al. (2024), cada observación se serializa como cadena de caracteres con precisión decimal fija. Por ejemplo, el valor 24.83 se convierte en la secuencia $\left\lbrack \text{'2'},\ \text{'4'},\ \text{'.'},\ \text{'8'},\ \text{'3'} \right\rbrack$. El *embedding* emplea una tabla de caracteres de vocabulario reducido —dígitos 0-9, punto decimal, signo negativo y espacio— con proyección lineal a $d_{model}$. Esta representación expande la secuencia original por un factor aproximado de 5x, generando múltiples *tokens* por cada observación. Se la denomina *text-based* en el resto del documento por brevedad, pero la implementación reproduce únicamente el mecanismo de tokenización por serialización de caracteres descrito por LLMTime, no la integración completa del modelo con un LLM de lenguaje natural. La Ilustración 4 muestra un ejemplo de serialización aplicada a la serie ETTh2.

<span id="_Ref226879662" class="anchor"></span>Ilustración 4. Tokenización text-based (LLMTime-inspired) sobre ETTh2: serialización decimal carácter a carácter.

<img src="./media/image6.png" style="width:6.29861in;height:1.86319in" />Fuente: Elaboración propia.

**Patching (***fixed-length segment tokenization***, PatchTST-inspired).** Siguiendo a Nie et al. (2023), la serie se segmenta en *patches* no solapados de longitud P *timesteps*. Cada *patch* se proyecta linealmente a un vector de dimensión $d_{model}$, reduciendo la longitud de la secuencia de $T\ a\ \frac{T}{P}$ *tokens*. Las dependencias intra-*patch* quedan capturadas implícitamente mediante la agregación, mientras que las dependencias inter-*patch* se modelan mediante la atención del *transformer*. La Ilustración 5 visualiza la segmentación en *patches* sobre la serie ETTh2, junto con la media, desviación y varianza de cada *patch*.

<span id="_Ref226879672" class="anchor"></span>Ilustración 5. Patching (PatchTST-inspired) sobre ETTh2: segmentación en patches no solapados y estadísticos por patch.

<img src="./media/image7.png" style="width:6.29861in;height:3.49028in" />Fuente: Elaboración propia.

**Descomposición (***trend–seasonal decomposition tokenization***, Autoformer-inspired).** Basada en la formulación de Wu et al. (2022), la serie se descompone en dos componentes aditivos mediante promedio móvil: tendencia y estacionalidad. Cada componente se proyecta por separado a dimensión $d_{model}$ y ambas representaciones se concatenan, generando *embeddings* que separan la dinámica de largo plazo de las fluctuaciones cíclicas. La Ilustración 6 muestra la descomposición resultante sobre ETTh2.

<span id="_Ref226879682" class="anchor"></span>Ilustración 6. Descomposición (Autoformer-inspired) sobre ETTh2: separación en componente de tendencia y estacional.

<img src="./media/image8.png" style="width:6.29861in;height:4.67222in" />Fuente: Elaboración propia.

**Foundation-inspired (***foundation-inspired masked patch tokenization***, MOMENT-inspired).** Siguiendo la arquitectura de Goswami et al. (2024), la serie se segmenta en *patches* y se enmascara aleatoriamente un porcentaje de ellos. Las posiciones enmascaradas se reemplazan por un *mask token* aprendible, y todos los *patches* —visibles y enmascarados— se proyectan linealmente a $d_{model}$. Esta representación obliga al *transformer* a reconstruir los *patches* ausentes a partir del contexto, aprendiendo dependencias temporales de forma implícita. Se la denomina *foundation* en el resto del documento por brevedad, pero la implementación reproduce únicamente el mecanismo de tokenización con enmascaramiento de *patches*, no el pre-entrenamiento masivo de los *foundation models* originales (MOMENT, Chronos), que constituye un factor de confusión metodológicamente excluido del Plan A. La Ilustración 7 ilustra el proceso de enmascaramiento y el error de reconstrucción asociado.

<span id="_Ref226879744" class="anchor"></span>Ilustración 7. Tokenización foundation (MOMENT-inspired) sobre ETTh2: enmascaramiento aleatorio de patches y error de reconstrucción.

<img src="./media/image9.png" style="width:6.29861in;height:3.49028in" />Fuente: Elaboración propia.

**HMM (RITMO).** La propuesta central del TFG aplica el marco teórico desarrollado en las secciones 4.2.1 a 4.2.6 para tokenizar la serie mediante los estados ocultos de un HMM entrenado con Baum-Welch. El número de estados K se trata como hiperparámetro del HMM y se optimiza por *dataset* mediante barrido $K\ \epsilon\ \left\{ 3,\ 4,\ 5,\ 6,\ 7,\ 8,\ 9,\ 10 \right\}$, seleccionando el valor que minimiza el MSE de predicción en validación. A partir del HMM entrenado, se implementan tres variantes de tokenización que representan niveles crecientes de preservación de información:

*Viterbi hard*: cada *timestep* se asigna al estado más probable mediante el algoritmo de Viterbi (sección 4.2.4) y recibe el *embedding* fijo $e_{k}$ definido en la sección 4.2.6. Los *timesteps* asignados al mismo estado comparten *embedding* idéntico, lo que produce compresión mediante *run-length encoding* —segmentos consecutivos del mismo estado se representan como un único *token* con su duración—.

*Soft (gamma)*: en lugar de una asignación determinística, cada *timestep* recibe un *embedding* ponderado $e_{t} = \sum_{k}^{}{\gamma_{t}(k)} \cdot e_{k}$, donde $\gamma_{t}(k)\ $son las probabilidades *a posteriori* obtenidas mediante el algoritmo *Forward-Backward* (sección 4.2.2). Esta formulación preserva la incertidumbre en la asignación de estados, produciendo un *embedding* continuo y único por *timestep*.

*Soft residual*: la variante más completa añade un residual intra-régimen $r_{t} = \frac{x_{t} - \mu_{k}}{\sigma_{k}}\ $que codifica la posición de la observación dentro del régimen asignado, construyendo $e_{t} = \left\lbrack r_{t},\sum_{k}^{}{\gamma_{t}(k)} \cdot e_{k} \right\rbrack$. El residual preserva la información continua que las dos variantes anteriores descartan al reducir cada observación a los parámetros de su régimen.

De las tres variantes anteriores, el Plan A (sección 5.1.1) evalúa únicamente *soft* y *soft residual*. La variante *hard* se descarta *a priori* porque la asignación determinística introduce un cuello de botella discreto que amortigua los gradientes y sesga las predicciones hacia la media del régimen dominante (Mensch & Blondel, 2018); un *head-to-head* formal frente a las variantes *soft* queda identificado como dimensión de sensibilidad no cubierta en este trabajo.

<span id="_Toc229829349" class="anchor"></span>Ilustración 8. Tokenización HMM (K = 5) sobre ETTh2: asignación de estados, secuencia de tokens, matriz de transición y ocupación.

<img src="./media/image10.png" style="width:6.29861in;height:2.77014in" />Fuente: Elaboración propia.

La tercera variante puede interpretarse como una normalización jerárquica: RevIN normaliza la ventana global eliminando diferencias de escala entre instancias, y el HMM normaliza cada *timestep* dentro de su régimen local, proporcionando al *transformer* información tanto sobre el contexto del régimen como sobre la posición relativa de la observación en él.

La Tabla 1 resume las seis técnicas en términos de método, tipo de *embedding* generado, número de *tokens* por observación y ratio de compresión resultante.

<span id="_Ref226879852" class="anchor"></span>Tabla 1. Comparativa formal de las seis técnicas de tokenización implementadas

| Técnica        | Método                              | Embedding                                                                     | Tokens/obs.     | Compresión                 |
|----------------|-------------------------------------|-------------------------------------------------------------------------------|-----------------|----------------------------|
| Discretización | Cuantización $(K = 8)$              | Tabla aprendible $\left\lbrack K,d_{model} \right\rbrack$                     | $$1$$           | $$1x$$                     |
| *Text-based*   | Serialización caracteres            | *Char embeddings*                                                             | $$\sim 5$$      | $$0.2x\ (expansión)$$      |
| *Patching*     | Segmentación $(P = 16)$             | Proyección lineal $\left\lbrack P,d_{model} \right\rbrack$                    | $$\frac{1}{P}$$ | $$Px$$                     |
| Descomposición | Promedio móvil                      | Proyección por componente                                                     | $$\frac{2}{T}$$ | $$\frac{T}{2}\ (extrema)$$ |
| *Foundation*   | *Patches + masking* $(mask = 30\%)$ | *Patch + mask token*                                                          | $$\frac{1}{P}$$ | $$Px$$                     |
| HMM (RITMO)    | Viterbi / *Forward-Backward*        | $$e_{k} = \left\lbrack \mu_{k},\sigma_{k},A\lbrack k,:\rbrack \right\rbrack$$ | $$1$$           | $$1x$$                     |

Fuente: Elaboración propia.

Estas seis técnicas constituyen el universo de comparación controlada del Plan A. El Plan B (sección 5.1.2) se restringe a la mejor variante de RITMO por *dataset* —ahora denominada RITMO-M y aplicada en modo *channel-independent* (sección 4.2.1)— frente a los cuatro *baselines* del estado del arte (DLinear, PatchTST, TimeMixer y TimeXer) en su configuración publicada; las cinco alternativas determinísticas no se reproducen en Plan B porque su finalidad era aislar el efecto de la tokenización con un *backbone* común, condición ya verificada en Plan A.

## Modelo de predicción: Transformer común

La quinta y última fase del *pipeline* RITMO es el modelo de predicción que recibe los *embeddings* generados por la fase de tokenización y produce la predicción del horizonte O. Para garantizar la comparación controlada definida en la sección 5.1.1 (Plan A), todas las técnicas de tokenización comparten un único *backbone transformer* —denominado *TransformerCommon*— cuyos hiperparámetros se mantienen idénticos en los experimentos. Esta arquitectura común aísla el efecto de la tokenización: cualquier diferencia de MSE entre técnicas es atribuible a la representación, no al modelo de predicción. El mismo *backbone* se reutiliza sin modificaciones en RITMO-M (Plan B, sección 5.1.2), cambiando únicamente las dimensiones de entrada y salida ($\text{enc}_{\text{in}} = c_{\text{out}} = C$ en lugar de 1) para procesar los C canales del *dataset* simultáneamente.

El diseño sigue la línea *encoder-only* de PatchTST (Nie et al., 2023), que demostró que un *encoder* sin *decoder* es suficiente para *forecasting* a largo plazo. Cada capa del *encoder* consta de dos sub-bloques con conexiones residuales y normalización de capa *pre-norm*: un módulo de *multi-head self-attention* con atención completa sin máscara causal —cada posición atiende a toda la secuencia de entrada y la dimensión de cada cabeza es $\frac{d_{model}}{n_{heads}}$ — y una *feed-forward network* de dos capas lineales con activación GELU y *dropout* intermedio cuya dimensión interna $d_{ff}$ controla la capacidad expresiva de cada posición. La codificación posicional emplea funciones sinusoidales (Vaswani et al., 2017), determinísticas y extrapolables a longitudes no vistas durante el entrenamiento. La cabeza de salida aplana la representación del *encoder* —concatenando las representaciones de todas las posiciones— y proyecta linealmente al horizonte de predicción, de modo que cada *timestep* predicho es función de toda la secuencia de entrada.

La integración de las seis técnicas de tokenización con este *backbone* común sigue un flujo de seis pasos compartidos por todas: (i) carga del *batch* de series temporales $\lbrack B,I,1\rbrack$ desde el *data loader*; (ii) aplicación de RevIN almacenando las estadísticas $(\mu,\ \sigma)$; (iii) tokenización según la técnica seleccionada (sección 4.2.7); (iv) generación de *embeddings* de dimensión $\left\lbrack B,\ seq,\ d_{model} \right\rbrack$ mediante el módulo correspondiente; (v) *forward pass* del *transformer* $\left\lbrack B,\ seq,\ d_{model} \right\rbrack\  \rightarrow \ \lbrack B,\ O,\ 1\rbrack$ y desnormalización de la predicción con RevIN⁻¹ $ŷ = \ ŷ_{norm} \bullet \ \sigma + \ \mu$; (vi) cálculo del MSE entre predicción y *ground truth* en la escala original. Las técnicas con longitud de secuencia variable —*text-based*, *patching* y *foundation*— se adaptan mediante *adaptive average pooling* a la longitud estándar ${seq}_{len} = 96\ $antes de alimentar el *encoder*; esta operación es agnóstica a la técnica y no introduce parámetros adicionales. Para las variantes HMM que requieren *Forward-Backward* (*soft*, *soft residual*), las posteriores γ se calculan en NumPy sobre el *batch* completo antes de pasar a PyTorch para el *embedding* y el *forward pass*. Los parámetros HMM $\left( A,\ \mu_{k},\ \sigma_{k},\ \pi \right)$ se cargan desde la caché persistente y permanecen fijos; solo la proyección lineal de *embeddings* y los parámetros del *transformer* se actualizan durante el entrenamiento.

<span id="_Ref226962836" class="anchor"></span>Ilustración 9. Flujo de integración del *TransformerCommon* con las seis técnicas de tokenización: diagrama de bloques de los seis pasos del *forward pass* y sus dimensiones tensoriales.

<img src="./media/image11.png" style="width:6.29861in;height:3.25in" />Fuente: Elaboración propia.

La Ilustración 9 sintetiza este flujo de integración en forma de diagrama de bloques: cada uno de los seis pasos aparece como un módulo conectado por flechas que representan el flujo de datos y sus dimensiones tensoriales, lo que permite seguir visualmente cómo cada técnica de tokenización se acopla al *backbone* común sin alterar su arquitectura interna. El detalle de los hiperparámetros del *TransformerCommon* —incluidos los intervalos explorados durante la búsqueda empírica y la justificación de cada decisión— se documenta en la sección 5.7 (Tabla 19).

## Herramientas y tecnologías

El desarrollo se realiza en Python 3.10 sobre un entorno Conda. El framework de deep learning empleado es PyTorch 2.9.0, utilizado tanto para la implementación del módulo HMM como para los modelos transformer y los baselines. Los experimentos del Plan A se ejecutan en CPU local; el Plan B —con cuatro baselines SOTA en modo --features M más el K-sweep de 256 corridas RITMO-M— se ejecuta en infraestructura GPU separada (RTX 3090 Ti en RunPod) por su mayor coste computacional. Las operaciones numéricas y estadísticas se apoyan en NumPy, Pandas, SciPy y scikit-learn —esta última empleada para la inicialización k-means del HMM—. La visualización emplea Matplotlib con paleta Okabe-Ito para accesibilidad en personas con daltonismo. La manipulación de tensores y las operaciones de reshape para patching emplean la biblioteca einops.

El código se construye sobre Time-Series-Library (TSLib, <https://github.com/thuml/Time-Series-Library>), repositorio de código abierto que proporciona implementaciones de referencia de los cuatro baselines del Plan B (DLinear, PatchTST, TimeMixer, TimeXer), el framework experimental con rutinas de entrenamiento, validación y test, y los data loaders para los seis datasets del estudio. Sobre esta base se integran los módulos propios del TFG: el módulo HMM para el entrenamiento Baum-Welch y la decodificación Viterbi, el generador de embeddings estructurados, las implementaciones de las seis técnicas de tokenización y la normalización reversible. Los parámetros HMM entrenados se almacenan en caché persistente para garantizar reproducibilidad exacta entre ejecuciones. El código fuente completo, junto con instrucciones de instalación y ejecución, está disponible en el repositorio público del proyecto: <https://github.com/jaime-oriol/RITMO>.

## Datasets e ingeniería del dato

La evaluación empírica emplea seis *datasets* de series temporales procedentes de los *benchmarks* consolidados de TSLib, divididos en dos grupos según su papel en el diseño experimental. La Tabla 2 resume sus características principales.

<span id="_Ref226879964" class="anchor"></span>Tabla 2. Datasets empleados en la evaluación experimental

| Dataset     | Dominio      | Frecuencia | Observaciones | Variables | Target (OT)                             |
|-------------|--------------|------------|---------------|-----------|-----------------------------------------|
| ETTh1       | Energía      | Horaria    | 17.420        | 7         | Temperatura aceite transformador        |
| ETTh2       | Energía      | Horaria    | 17.420        | 7         | Temperatura aceite (variante con shift) |
| Weather     | Meteorología | 10 min     | 52.696        | 21        | Sensor MPI Biogeochemistry              |
| Electricity | Energía      | Horaria    | 26.304        | 321       | Consumo eléctrico (cliente 320)         |
| Traffic     | Transporte   | Horaria    | 17.544        | 862       | Ocupación sensor de tráfico             |
| Exchange    | Finanzas     | Diaria     | 7.588         | 8         | Tipo de cambio agregado                 |

Fuente: Elaboración propia con datos de TSLib (<https://github.com/thuml/Time-Series-Library>). Los datasets originales provienen de Zhou et al. (2021) para ETTh1 y ETTh2, Lai et al. (2018) para Traffic y Exchange, el Max-Planck-Institute for Biogeochemistry para Weather y el UCI Machine Learning Repository para Electricity.

Los cuatro primeros *datasets* constituyen el **Grupo 1 (entrenamiento HMM)**, sobre los cuales se entrenan los parámetros λ del modelo y se ejecutan los experimentos de predicción. ETTh1 y ETTh2 son variantes del *benchmark* Electricity Transformer Temperature (H. Zhou et al., 2021) que registran la temperatura del aceite de un transformador eléctrico; la segunda variante presenta *distribution shift* más pronunciado, lo que permite evaluar la robustez de la normalización RevIN. Weather contiene lecturas meteorológicas con regímenes estacionales claros, idóneo para verificar que el HMM captura cambios de régimen. Electricity registra el consumo eléctrico de un cliente con periodicidad diaria y semanal pronunciada, adecuado para evaluar la persistencia de los *tokens* HMM frente a ciclos regulares. Los dos restantes forman el **Grupo 2 (transferencia** *cross-domain* **del** *tokenizer* **HMM congelado)**, donde se aplica un HMM entrenado en otro *dataset* sin re-entrenamiento de parámetros: Traffic (Lai et al., 2018) registra la ocupación de sensores de tráfico con regímenes *rush-hour* explícitos, y Exchange (Lai et al., 2018) recoge tipos de cambio diarios sin periodicidad marcada, constituyendo un test de robustez extremo para un modelo diseñado para capturar regímenes. La definición precisa de este protocolo —referido por brevedad como evaluación *zero-shot*— se introduce en la sección 5.1.1.

Las cuatro subsecciones siguientes documentan las decisiones de ingeniería del dato adoptadas en este trabajo. Dado que los seis *datasets* son *benchmarks* públicos consolidados —no datos recolectados *ad hoc*— el rol de la ingeniería del dato no consiste en construir un *pipeline* ETL desde cero, sino en seleccionar fuentes comparables con la literatura, integrarlas a través de un *loader* común, validarlas y aplicar las transformaciones requeridas por el modelo.

## Origen, obtención y criterio de selección

Los seis *datasets* provienen de repositorios públicos consolidados que constituyen el estándar de evaluación en el campo del *long-term forecasting*. La obtención se realiza mediante el módulo `data_provider` del repositorio Time-Series-Library (TSLib), que automatiza la descarga y parseo de los archivos CSV originales y expone una interfaz uniforme —`Dataset_ETT_hour` y `Dataset_Custom`— para cada familia de *dataset*. El criterio de selección responde a tres exigencias metodológicas: (i) **comparabilidad con la literatura**, ya que estos seis *datasets* son los empleados por DLinear (Zeng et al., 2023), PatchTST (Nie et al., 2023), TimeMixer (S. Wang et al., 2024), TimeXer (Y. Wang et al., 2024) y la práctica totalidad de los trabajos de *long-term forecasting* publicados en los últimos cinco años, lo que permite contrastar los resultados del Plan B sin reentrenar los *baselines*; (ii) **diversidad de dominios y dinámicas**, cubriendo desde patrones físicos altamente regulares hasta señales financieras casi aleatorias, lo que somete al HMM a regímenes estadísticos cualitativamente distintos; y (iii) **separación natural entre entrenamiento y transferencia** *cross-domain* **del** *tokenizer*, reservando Traffic y Exchange como prueba de transferencia del HMM a dominios no vistos.

## Estructura, validación e integridad

Para garantizar que los seis *datasets* cumplen los requisitos exigidos por el *pipeline* RITMO se ejecuta un análisis exploratorio reproducible y cuya versión visual con gráficos se documenta en el Anexo A. La Tabla 3 resume los descriptores estructurales obtenidos.

<span id="_Ref226880013" class="anchor"></span>Tabla 3. Estructura e integridad de los seis *datasets*

| Dataset     | Shape        | Vars | Tipo    | Nulls (isnull) | Sentinels (–9999) |
|-------------|--------------|------|---------|----------------|-------------------|
| ETTh1       | (17420, 8)   | 7    | float64 | 0              | 0                 |
| ETTh2       | (17420, 8)   | 7    | float64 | 0              | 0                 |
| Weather     | (52696, 22)  | 21   | float64 | 0              | 50 (0.095 %)      |
| Electricity | (26304, 322) | 321  | float64 | 0              | 0                 |
| Traffic     | (17544, 863) | 862  | float64 | 0              | 0                 |
| Exchange    | (7588, 9)    | 8    | float64 | 0              | 0                 |

Fuente: Elaboración propia

Tres propiedades quedan verificadas. **Integridad temporal**: los seis archivos presentan un índice regular sin huecos en su cadencia nominal (horaria, cada 10 minutos o diaria), por lo que no se requiere imputación ni interpolación. **Tipos homogéneos**: todas las variables son numéricas `float64` y la columna temporal se convierte explícitamente a `datetime`. **Ausencia de valores nulos**: el método `pandas.isnull()` reporta 0 ocurrencias en los seis *datasets*, si bien en Weather se identifican manualmente 50 valores sentinel `-9999` (0.095 % del total) que constituyen *missing values* encubiertos según la convención del Max-Planck-Institute for Biogeochemistry. Estos valores se mantienen tal cual durante el entrenamiento porque su volumen es despreciable y la normalización RevIN absorbe su efecto a nivel de instancia. La decisión de **no eliminar valores extremos** es deliberada: las series provienen de sensores físicos reales en los que los valores atípicos son eventos legítimos —olas de calor, picos de demanda, congestiones, crisis cambiarias— cuya eliminación distorsionaría la señal que se desea modelar.

Respecto al particionado, verificado en `data_provider/data_loader.py`, el `Dataset_ETT_hour` aplica la división canónica de Zhou et al. (2021) en 12 meses de entrenamiento, 4 meses de validación y 4 meses de test (8 640 / 2 880 / 2 880 *timesteps* horarios), mientras que `Dataset_Custom` emplea la división 70 % / 10 % / 20 % para Weather, Electricity, Traffic y Exchange. En todos los casos las particiones son estrictamente cronológicas, garantizando que la evaluación replica la situación real de predicción sobre el futuro.

## Análisis estadístico y temporal

La Tabla 4 sintetiza los descriptores estadísticos del *target* y los resultados del test Augmented Dickey-Fuller (ADF) aplicado sobre la serie completa, junto con el análisis de autocorrelación y el *distribution shift* entre las particiones train y test. Los datos provienen del mismo *script* reproducible referenciado en la subsección anterior.

<span id="_Ref226880033" class="anchor"></span>Tabla 4. Análisis estadístico y temporal del target en cada *dataset*

| Dataset     | Rango (min – max) | μ (std)        | ADF p-val | Estac. | Pico ACF dominante | Δμ train→test |
|-------------|-------------------|----------------|-----------|--------|--------------------|---------------|
| ETTh1       | –4.08 – 46.01     | 13.33 (8.57)   | 8.3·10⁻³  | Sí     | lag 24 (diario)    | 71.7 %        |
| ETTh2       | –2.65 – 58.88     | 26.61 (11.89)  | 5.8·10⁻³  | Sí     | lag 24 (diario)    | 46.4 %        |
| Weather     | 305.5 – 524.2     | 427.69 (—)     | \< 10⁻²⁰  | Sí     | lag 144 (diario)   | 2.8 %         |
| Electricity | 0 – 6035          | 3335.9 (552.7) | 3·10⁻²⁶   | Sí     | lag 168 (semanal)  | 0.4 %         |
| Traffic     | 0 – 0.217         | 0.032 (0.019)  | 4·10⁻²¹   | Sí     | lag 168 (semanal)  | 25.6 %        |
| Exchange    | 0.393 – 0.882     | 0.654 (0.115)  | 0.42      | No     | sin pico claro     | 26.4 %        |

Fuente: Elaboración propia. Para Weather las estadísticas excluyen los 50 valores sentinel `-9999`. ADF aplicado sobre la serie completa sin submuestreo; H₀ rechazada (serie estacionaria) si p \< 0.05.

El análisis revela tres hallazgos relevantes para el diseño metodológico. **Estacionariedad mayoritaria**: cinco de los seis *datasets* —incluidos los cuatro del Grupo 1— rechazan la hipótesis de raíz unitaria con p-values inferiores a 0.01. Únicamente Exchange presenta no estacionariedad clara (p = 0.42), consistente con su naturaleza de tipos de cambio sin tendencia ni periodicidad. **Periodicidades dominantes verificadas mediante ACF**: ETTh1, ETTh2 y Weather presentan ciclo diario (lag 24 en *datasets* horarios y lag 144 en Weather, muestreado cada 10 min), Electricity y Traffic exhiben un ciclo *semanal* (lag 168) más fuerte aún que el diario, coherente con sus regímenes de uso humano, y Exchange no presenta picos significativos en la ACF. **Distribution shift acusado en cuatro** *datasets*: el cambio relativo de la media entre las particiones train y test supera el 10 % en ETTh1 (71.7 %), ETTh2 (46.4 %), Traffic (25.6 %) y Exchange (26.4 %), confirmando empíricamente la necesidad de la normalización por instancia descrita en la sección 4.4.4. Esta combinación de regímenes claros, periodicidades persistentes y *distribution shift* acotable confirma la idoneidad del marco HMM para los cuatro *datasets* del Grupo 1, mientras que el caso de Exchange anticipa que la transferencia *cross-domain* del *tokenizer* HMM sobre este *dataset* será el escenario más exigente del Plan A.

La visualización gráfica completa del análisis exploratorio se reporta en el **Anexo A** al final del documento e incluye, para cada uno de los seis *datasets*, las series temporales con los cortes de las particiones, los histogramas de distribución del *target*, las funciones de autocorrelación hasta lag 400 y la comparación visual del *distribution shift* entre train y test.

## Transformaciones de preprocesamiento aplicadas

La fase de transformación se aplica en dos etapas complementarias. La primera la realiza el propio `data_provider` mediante un `StandardScaler` (Pedregosa et al., 2011) ajustado **únicamente** sobre la partición de entrenamiento y aplicado a las tres particiones, garantizando que ninguna estadística de validación o test contamina el ajuste del modelo. La segunda etapa la realiza el módulo RevIN dentro del modelo (sección 4.2.5), que normaliza cada **instancia** de longitud $I = 96$ *timesteps* almacenando su media y desviación locales para revertirlas sobre la predicción. Esta doble normalización es necesaria porque las series presentan el *distribution shift* documentado en la Tabla 4: el escalado global por sí solo no es suficiente para que las estadísticas de cada instancia coincidan con las del entrenamiento, mientras que RevIN sin un escalado global previo dejaría al HMM operando sobre rangos heterogéneos entre *datasets*.

Por último, los dos planes experimentales se ejecutan con modos de *features* distintos. El **Plan A** emplea el modo *features* S (univariado puro), donde el modelo recibe únicamente la variable objetivo como canal de entrada —la columna OT en los seis *datasets*—. La elección es deliberada y responde a la necesidad de aislar el efecto de la tokenización: al limitar la entrada del modelo a una única serie, cualquier diferencia de desempeño entre las seis técnicas comparadas es atribuible exclusivamente a la representación, eliminando señales auxiliares que podrían enmascarar el efecto que se desea medir. El **Plan B**, en cambio, emplea *features* M con *channel-independence* siguiendo la convención de la literatura SOTA: los *baselines* (DLinear, PatchTST, TimeMixer, TimeXer) publican sus resultados en este modo, por lo que reproducirlo es condición necesaria para una comparación equitativa; en este modo el HMM se entrena sobre los C canales del *dataset* (sección 4.2.1) y el *transformer* recibe y predice las C series simultáneamente. Esta asimetría es intencional: cada plan opera en sus condiciones óptimas —Plan A para la comparación controlada, Plan B para la paridad publicada con el estado del arte—.

## Métricas de evaluación

La evaluación del sistema emplea dos familias complementarias de métricas: métricas intrínsecas, que caracterizan la calidad de la tokenización independientemente de la tarea *downstream*, y métricas de predicción, que cuantifican el desempeño en *forecasting* a largo plazo.

## Métricas intrínsecas de tokenización

Se definen diez métricas organizadas en tres grupos según su aplicabilidad. Las **métricas universales** se aplican a las seis técnicas:

**1. Ratio de compresión.** Relación entre *timesteps* originales y *tokens* generados $\left( \frac{T}{N_{tokens}} \right)$. Cuantifica el grado de compactación de la representación: valores altos indican mayor síntesis, mientras que valores inferiores a 1 señalan expansión de la secuencia. Esta métrica se utiliza como medida básica de eficiencia en la mayoría de los trabajos sobre tokenización de series temporales (Ansari et al., 2024; Talukder et al., 2025).

**2. Error de reconstrucción (MSE).** Error cuadrático medio al reconstruir la serie original desde los *tokens*. Valores cercanos a cero indican tokenización *lossless*; valores altos señalan pérdida de información irreversible. La reconstrucción se emplea habitualmente como métrica intrínseca de calidad en *autoencoders* y modelos de cuantización vectorial (Oord et al., 2018).

**3. Retención de autocorrelación.** Correlación de Pearson entre la función de autocorrelación (ACF, *Autocorrelation Function*) de la serie original y la de la serie reconstruida, calculada hasta lag 20. Evalúa si la tokenización preserva la estructura de dependencias temporales de la serie. Esta métrica adapta las medidas clásicas de validación de modelos lineales (Box & Jenkins, 1976) al contexto de evaluación de representaciones discretas.

Las **métricas discretas** se aplican exclusivamente a las técnicas que generan vocabulario finito —discretización y HMM—:

**4. Entropía de vocabulario.** Entropía de Shannon normalizada sobre la distribución de unigramas. Valores altos indican uso equilibrado del vocabulario disponible; valores bajos señalan dominancia de unos pocos *tokens*.

**5. Entropía de bigramas.** Entropía de las transiciones entre *tokens* consecutivos. Valores bajos indican patrones de transición predecibles, lo que en el caso del HMM refleja la estructura impuesta por la matriz A.

**6. Persistencia de tokens.** Longitud media de *runs* consecutivos del mismo *token*. Captura la capacidad de la técnica para identificar regímenes estables: valores altos indican que los *tokens* representan segmentos coherentes de la serie. Esta medida es habitual en la evaluación de modelos de segmentación basados en HMM (Rabiner, 1989).

**7. Cobertura top-k.** Fracción del uso total cubierta por los k *tokens* más frecuentes. Evalúa la concentración del vocabulario, útil para detectar distribuciones degeneradas.

Las **métricas de robustez** evalúan la estabilidad de la representación cuando la serie original se perturba con ruido gaussiano de desviación estándar $\sigma \in \text{\{}0.1 \cdot \sigma_{X},\, 0.5 \cdot \sigma_{X}\text{\}}$ —donde $\sigma_{X}$ es la desviación de la propia serie— y se re-tokeniza:

**8. Cambio relativo en MSE de reconstrucción.** Variación porcentual del MSE de reconstrucción al re-tokenizar la serie perturbada respecto a la original. Captura la fragilidad de la representación frente a ruido aditivo: valores próximos a cero indican estabilidad, valores altos indican que la tokenización es muy sensible a perturbaciones de pequeña amplitud.

**9. Edit distance normalizada (técnicas discretas).** Distancia de Levenshtein entre la secuencia de *tokens* de la serie original y la perturbada, normalizada por la longitud de la mayor de las dos. Cuantifica el cambio estructural inducido por la perturbación en el espacio discreto de *tokens*.

**10. L₂ media en token-space (técnicas continuas).** Distancia euclídea media entre los *tokens* emparejados —*patches* o componentes— de la serie original y la perturbada, computada en el espacio de *tokens* definido por cada técnica. Constituye el equivalente continuo de la edit distance.

Los valores numéricos de estas diez métricas sobre los cuatro *datasets* del Grupo 1, agregados sobre 3 *seeds*, se presentan en la sección 5.6 junto con la comparativa intrínseca de las seis técnicas; su análisis cualitativo y la discusión de los hallazgos se desarrollan en el Capítulo 7.

## Métricas de predicción

El desempeño *downstream* se evalúa mediante las dos métricas estándar del protocolo TSLib establecido por Zhou et al. (2021):

> $$MSE = \frac{1}{N}\sum_{t = 1}^{N}\left( y_{t} - \widehat{y_{t}} \right)^{2}$$
>
> $$MAE = \frac{1}{N}\sum_{t = 1}^{N}\left| y_{t} - \widehat{y_{t}} \right|$$

Donde $y_{t}$ e $ŷ_{t}$ son los valores reales y predichos respectivamente, evaluados sobre el conjunto de test. Ambas métricas se calculan tras la desnormalización RevIN, es decir, en la escala original de los datos. Se reportan por horizonte individual y como promedio sobre los cuatro horizontes $O\  \in \text{\{}96,\ 192,\ 336,\ 720\text{\}}$, siguiendo la convención adoptada en los principales *benchmarks* del campo (Nie et al., 2023; S. Wang et al., 2024; Y. Wang et al., 2024; Wu et al., 2022; Zeng et al., 2023)

El protocolo de reporte difiere por plan. Para el **Plan A**, todas las cifras son agregados mean ± std sobre las 3 *seeds* {42, 2021, 7} descritas en la sección 4.7, y las comparaciones entre el tokenizador HMM y cada uno de los cinco *baselines* deterministas se contrastan mediante el test pareado de Wilcoxon de signo-rango sobre las 12 observaciones por *dataset* (4 horizontes × 3 *seeds*), con corrección Bonferroni para las cinco comparaciones al nivel de significación α = 0.01. Para el **Plan B**, ejecutado con seed única (2021) por restricción de presupuesto GPU, los valores se reportan como mediciones puntuales y no admiten test estadístico formal; las comparaciones entre RITMO-M y los cuatro *baselines* SOTA se presentan en términos descriptivos (rankings, márgenes relativos y conteo de victorias por horizonte).

## Protocolo multi-seed y selección de K

Para evitar artefactos de un único punto de inicialización en la selección de K o en la inicialización del *transformer*, **el Plan A** se ejecuta bajo un protocolo multi-seed con tres *seeds* {42, 2021, 7}. Cada *seed* controla cuatro fuentes de estocasticidad: (i) la inicialización *k-means* del HMM, (ii) el muestreador del enmascarado de la técnica MOMENT-inspired, (iii) el muestreador de ruido de las métricas de robustez (sección 4.5.1) y (iv) la inicialización Xavier de los pesos del *transformer*. Los *baselines* determinísticos —SAX-inspired, LLMTime-inspired, PatchTST-inspired y Autoformer-inspired— producen std ≈ 0 por construcción en las métricas intrínsecas, pero igualmente se entrenan y evalúan bajo las tres *seeds* para homogeneizar las cifras *downstream*.

La selección del número de estados K del HMM se realiza en dos fases. En la **fase 1** se entrena un HMM para cada combinación de *dataset* y $K\  \in \text{\{}3,\ 4,\ 5,\ 6,\ 7,\ 8,\ 9,\ 10\text{\}}$, y para cada una de las tres *seeds*, generando 4 × 8 × 3 = **96 modelos cacheados**. En la **fase 2** se ejecuta el *pipeline* completo de predicción al horizonte $O\  = \ 96$ sobre cada caché con las dos variantes HMM (*soft* y *soft residual*), totalizando 4 × 8 × 2 × 3 = **192 ejecuciones controladas**. La configuración (variante, K) que minimiza el MSE de validación seed-promediado para cada *dataset* se toma como *óptimo robusto* y se reutiliza inalterada en los tres horizontes restantes $O\  \in \text{\{}192,\ 336,\ 720\text{\}}$.

Reusar el K seleccionado a $O\  = \ 96$ en horizontes largos sigue la práctica estándar de la literatura de *long-term forecasting* (Nie et al., 2023; S. Wang et al., 2024; Y. Wang et al., 2024; Wu et al., 2022): acoplar exhaustivamente el barrido de K con los cuatro horizontes multiplicaría el presupuesto de búsqueda por cuatro (768 ejecuciones controladas en lugar de 192) sin alterar el ranking cualitativo de configuraciones (variante, K) dentro de un *dataset*, dado que las estadísticas de régimen sobre las que opera la tokenización HMM se estiman una sola vez sobre la partición de entrenamiento y no dependen del horizonte de predicción. La implicación de que el K seleccionado a $O\  = \ 96$ podría no ser globalmente óptimo en horizontes largos se reconoce explícitamente como limitación en la sección 8.2.

La Tabla 5 resume la configuración (variante, K) seleccionada por *dataset*. Estas cuatro configuraciones son los únicos tokenizadores HMM utilizados en el resto del documento.

<span id="_Ref228520493" class="anchor"></span>Tabla 5. Óptimo robusto (variante, K) por dataset, seleccionado a partir de las 192 ejecuciones del barrido de K @ $O\  = \ 96$ (mean MSE sobre 3 seeds).

| Dataset     | Variante HMM      | K   |
|-------------|-------------------|-----|
| ETTh1       | hmm_soft          | 8   |
| ETTh2       | hmm_soft          | 9   |
| Weather     | hmm_soft_residual | 4   |
| Electricity | hmm_soft_residual | 3   |

Fuente: Elaboración propia.

La Ilustración 24 (Anexo B) visualiza el barrido completo: las 64 combinaciones (4 *datasets* × 2 variantes × 8 valores de K) con sus bandas de incertidumbre seed a seed, y la línea discontinua roja sobre el K seleccionado por *dataset* (Tabla 5). Los valores numéricos completos del barrido —MSE de validación seed a seed para las 64 combinaciones— se recogen en el mismo.

Para el **Plan B** se ejecuta un K-sweep análogo en modo *features* M con seed única (2021). El barrido recorre las mismas combinaciones $K \in \text{\{}3,\ 4,\ 5,\ 6,\ 7,\ 8,\ 9,\ 10\text{\}} \times \text{\{}\text{soft},\ \text{soft residual}\text{\}} \times 4\,\text{datasets}$ del Grupo 1, pero ahora extendido a los cuatro horizontes $O\  \in \text{\{}96,\ 192,\ 336,\ 720\text{\}}$—al disponer de cifras *downstream* en los cuatro horizontes, no es necesario reusar el K seleccionado a $O\  = \ 96\ $—, generando **32 caches HMM-M** y **256 ejecuciones controladas RITMO-M** (4 ds × 8 K × 2 variantes × 4 horizontes). El criterio de selección difiere consecuentemente del Plan A: la configuración (variante, K) óptima por *dataset* se elige como argmin del MSE de validación promediado sobre los 4 horizontes. La Tabla 6 resume el resultado.

<span id="_Ref228551187" class="anchor"></span>Tabla 6. Óptimo (variante, K) por dataset en RITMO-M (Plan B), seleccionado por argmin del MSE de validación promediado sobre los 4 horizontes (256 ejecuciones, seed = 2021).

| Dataset     | Variante HMM      | K   |
|-------------|-------------------|-----|
| ETTh1       | hmm_soft          | 5   |
| ETTh2       | hmm_soft          | 3   |
| Weather     | hmm_soft_residual | 5   |
| Electricity | hmm_soft_residual | 4   |

Fuente: Elaboración propia.

La Ilustración 25 (Anexo C) visualiza el K-sweep Plan B completo, análoga a la Ilustración 24 del Plan A: las 64 combinaciones (4 *datasets* × 2 variantes × 8 valores de K) con la línea discontinua roja sobre el K seleccionado por *dataset* (Tabla 6). Al ejecutarse con seed única, las curvas no incluyen bandas de incertidumbre.

La variante seleccionada coincide en los cuatro *datasets* con la del Plan A (Tabla 5) —hmm soft para ETT, hmm soft residual para Weather y Electricity—, validando que la doble especialización HMM observada en univariate se mantiene en multivariate *channel-independent*. El valor de K, en cambio, difiere en los cuatro *datasets* (8→5, 9→3, 4→5, 3→4): el HMM entrenado sobre los C canales apilados encuentra una granularidad óptima de regímenes distinta de la del HMM entrenado solo sobre OT. Los valores numéricos completos del K-sweep Plan B —MSE de validación seed única para las 64 combinaciones— se recogen en el Anexo C al final del documento, junto con la Ilustración 25.

# DESARROLLO TÉCNICO

Este capítulo detalla el diseño experimental y la implementación de cada fase del *pipeline* RITMO. La sección 5.1 formaliza los dos escenarios experimentales —Plan A y Plan B— que estructuran toda la evaluación posterior. Las secciones 5.2 a 5.5 documentan cómo se implementan las cuatro primeras fases del *pipeline* (RevIN, Baum-Welch, Viterbi y *embeddings* estructurados), qué decisiones técnicas se tomaron durante el desarrollo y qué resultados de validación se obtuvieron. La sección 5.6 sintetiza la comparativa intrínseca de las seis técnicas de tokenización aplicadas sobre los **cuatro *datasets* del Grupo 1**, agregada sobre 3 *seeds*, y la sección 5.7 cierra el capítulo describiendo la quinta fase —el modelo de predicción *transformer*— junto con la búsqueda y los intervalos de hiperparámetros explorados. Los fundamentos teóricos de cada algoritmo están formalizados en la sección 4.2 y no se repiten; se referencian cuando es necesario. La implementación completa está disponible en el repositorio del proyecto.

## Diseño experimental

El diseño experimental sigue dos escenarios complementarios que abordan preguntas distintas: el Plan A aísla el efecto de la tokenización mediante una comparación controlada de las seis técnicas formalizadas en la sección 4.2.7, mientras que el Plan B sitúa la propuesta en el contexto del estado del arte. Estos dos escenarios constituyen la estrategia de análisis experimental empleada para validar RITMO desde dos ángulos complementarios: la contribución específica de la tokenización en condiciones controladas y la competitividad global frente a arquitecturas especializadas.

## Plan A: comparación controlada de tokenizaciones

El Plan A evalúa las seis técnicas bajo condiciones experimentales idénticas, de modo que las diferencias en desempeño sean atribuibles exclusivamente a la representación. Todas las técnicas comparten el mismo *backbone transformer*, entrenado con el mismo optimizador, tasa de aprendizaje y número de *epochs*. La ventana de entrada se fija en $I = 96$ *timesteps* y los horizontes evaluados son $O\  \in \text{\{}96,\ 192,\ 336,\ 720\text{\}}$, siguiendo el protocolo estándar establecido por Zhou et al. (2021).

La evaluación combina las diez métricas intrínsecas —que caracterizan la calidad de cada tokenización antes de la predicción— con las métricas MSE y MAE de *forecasting*. Los cuatro *datasets* del Grupo 1 se evalúan con HMM entrenado sobre sus propios datos (288 ejecuciones controladas: 6 técnicas × 4 horizontes × 4 *datasets* × 3 *seeds*), mientras que los dos *datasets* del Grupo 2 se evalúan en modo de **transferencia** *cross-domain* **del** *tokenizer* **HMM congelado** —referido en el resto del documento como evaluación *zero-shot* en el sentido restringido de *frozen-tokenizer transfer*—, (216 ejecuciones: 9 fuentes de tokenizador × 4 horizontes × 2 *datasets* × 3 *seeds*) donde los parámetros del HMM se estiman en un *dataset* fuente y se aplican sin re-entrenamiento sobre el *dataset* objetivo, mientras que el *transformer* sí se entrena con los datos del *dataset* objetivo. Esta precisión terminológica es importante: no se trata de un *zero-shot* completo del sistema (en el que ningún parámetro se ajusta al dominio objetivo), sino de un protocolo intermedio en el que solo el *tokenizer* probabilístico es transferido de forma congelada.

El aporte de este escenario es la primera comparación sistemática de seis paradigmas de tokenización para series temporales en condiciones controladas, cuantificando los *trade-offs* entre compresión, preservación de información, captura de dependencias temporales y desempeño predictivo.

## Plan B: validación frente al estado del arte

El Plan B sitúa a **RITMO-M** —la versión del *pipeline* RITMO en modo --features M definida en la sección 4.1 y formalizada en 4.2.1— en el contexto de los modelos consolidados, en paridad con el régimen multivariado en que los cuatro *baselines* publican sus resultados. Los cuatro *baselines* empleados son DLinear (Zeng et al., 2023), PatchTST (Nie et al., 2023), TimeMixer (S. Wang et al., 2024) y TimeXer (Y. Wang et al., 2024), ejecutados con los hiperparámetros reportados en sus publicaciones originales y mediante los *scripts* proporcionados por TSLib. Esta configuración garantiza que cada *baseline* opera en sus condiciones óptimas, evitando penalizaciones por ajuste inadecuado. El K-óptimo de RITMO-M se selecciona por *dataset* mediante el K-sweep --features M descrito en la sección 4.7 (256 ejecuciones controladas, Tabla 6); con las 64 ejecuciones adicionales de los cuatro *baselines*, el Plan B totaliza **320 ejecuciones controladas**. Por restricciones de presupuesto computacional —tanto RITMO-M como los *baselines* corren sobre infraestructura GPU separada (RunPod RTX 3090 Ti)— la evaluación se realiza con seed única (2021) y restringida a los cuatro *datasets* del Grupo 1; la transferencia *cross-domain* sobre Traffic y Exchange queda fuera del alcance del Plan B y solo se reporta dentro del Plan A (sección 6.3).

La comparación directa permite cuantificar si la tokenización probabilística *channel-independent* —con el mismo TransformerCommon que el Plan A— aporta mejoras o pérdidas respecto a arquitecturas especializadas que integran su propia tokenización (el *patching* de PatchTST, la descomposición de DLinear/TimeMixer o la atención exógena de TimeXer) y que han sido objeto de años de optimización por parte de la comunidad. La Tabla 7 sintetiza las diferencias clave entre ambos escenarios.

<span id="_Ref226881947" class="anchor"></span>Tabla 7. Comparativa entre los dos escenarios del diseño experimental.

| Parámetro         | Plan A                                           | Plan B                                             |
|-------------------|--------------------------------------------------|----------------------------------------------------|
| Modelos evaluados | 6 técnicas + Transformer común                   | HMM+Transformer vs 4 baselines                     |
| Backbone          | Transformer común (único)                        | Arquitectura propia de cada modelo                 |
| Input length (I)  | 96                                               | 96                                                 |
| Horizontes (O)    | {96, 192, 336, 720}                              | {96, 192, 336, 720}                                |
| Datasets Grupo 1  | ETTh1, ETTh2, Weather, Electricity               | Mismos                                             |
| Datasets Grupo 2  | Traffic, Exchange (HMM transferido cross-domain) | No evaluado (presupuesto computacional)            |
| Métricas          | Intrínsecas (7) + MSE/MAE                        | MSE/MAE                                            |
| Hiperparámetros   | Idénticos para las 6 técnicas                    | Óptimos reportados en papers                       |
| Seeds             | 3 {42, 2021, 7}                                  | 1 (presupuesto computacional, infraestructura GPU) |

Fuente: Elaboración propia.

Quedan fuera del alcance experimental los *foundation models* completos con pre-entrenamiento masivo —MOMENT pre-entrenado, Chronos— por introducir un factor de confusión (los datos de pre-entrenamiento) que impediría aislar el efecto de la tokenización. Asimismo, se excluyen TOTEM y VQ-VAE en su configuración original por estar diseñados para escenarios multivariados sin *benchmarks* univariados publicados. Las implementaciones de *foundation* y *discretización* incluidas en el Plan A reproducen el mecanismo de tokenización de estos modelos, no su pre-entrenamiento, lo que permite evaluar la técnica de representación en igualdad de condiciones.

## Fase 1: Normalización RevIN

La normalización RevIN se implementa como un módulo que expone dos operaciones: normalización con almacenamiento de estadísticas, y desnormalización para restaurar la escala original. El módulo calcula media $\mu$ y desviación $\sigma$ sobre la partición de entrenamiento y aplica la transformación definida en la sección 4.2.5 a cada instancia de forma independiente.

A modo de ejemplo, la validación sobre ETTh2 confirma el comportamiento esperado. La serie original presenta $\mu = \ 28.82\ y\ \sigma = \ 11.40$. La normalización se aplica por ventanas de 96 *timesteps*, y tras la transformación las estadísticas resultantes son $\mu \approx 0.0\ y\ \sigma \approx 1.0$, verificando que la operación centra y escala correctamente cada ventana. La prueba de reversibilidad —normalizar y desnormalizar secuencialmente— produce un MSE de reconstrucción inferior a $10^{- 12}$ ($MSE < 10^{- 12}$), confirmando que la operación es numéricamente *lossless*.

<span id="_Toc229829351" class="anchor"></span>Ilustración 10. Validación de RevIN sobre ETTh2: serie original (μ = 28.82, σ = 11.40) y serie tras normalización per-window (μ ≈ 0, σ ≈ 1).

<img src="./media/image12.png" style="width:6.29861in;height:2.67431in" />Fuente: Elaboración propia.

Este resultado es relevante porque garantiza que cualquier error introducido en las fases posteriores del *pipeline* es atribuible exclusivamente a la tokenización o al modelo de predicción, no a la normalización.

## Fase 2: Entrenamiento HMM (Baum-Welch)

El algoritmo de Baum-Welch se implementa con todas las operaciones en espacio logarítmico para evitar *underflow* numérico con secuencias largas. La aritmética en log-espacio se centraliza mediante una función de normalización logarítmica que aplica el truco *log-sum-exp*, manteniendo estabilidad numérica durante las pasadas *forward* y *backward* incluso con secuencias de más de 10 000 *timesteps*.

La inicialización de parámetros emplea *k-means* sobre las observaciones normalizadas, lo que proporciona centros iniciales $\mu_{k}$ y desviaciones $\sigma_{k}$ razonables para cada estado. La distribución inicial $\pi$ se fija uniforme y la matriz de transición A se inicializa uniforme con ruido aleatorio pequeño, permitiendo que el algoritmo EM descubra la estructura de transiciones sin imponer sesgo inicial. El criterio de convergencia es $|\Delta LL| < \varepsilon = 10^{- 4}$ donde ΔLL es el incremento de log-verosimilitud entre iteraciones consecutivas.

El número de estados K es un hiperparámetro del sistema que se optimiza por *dataset* mediante barrido $K\  \in \text{\{}3,\ 4,\ 5,\ 6,\ 7,\ 8,\ 9,\ 10\text{\}}$ descrito en la sección 4.7, seleccionando para cada *dataset* y cada variante (*soft*, *soft residual*) el valor que minimiza el MSE de validación seed-promediado al horizonte $O\  = \ 96\ $(192 ejecuciones controladas; valores numéricos completos en el Anexo B). El barrido genera **96 modelos cacheados** —8 K × 4 *datasets* × 3 *seeds*— y revela que el óptimo robusto varía entre *datasets* (Tabla 5): K = 8 *soft* para ETTh1, K = 9 *soft* para ETTh2, K = 4 *soft residual* para Weather y K = 3 *soft residual* para Electricity. La variación refleja diferencias en la estructura temporal de cada serie: ETTh1 y ETTh2 —series volátiles con *distribution shift*— se benefician de un vocabulario relativamente amplio (K alto) que absorba la heterogeneidad de la distribución; Weather y Electricity —regímenes persistentes con ciclos diarios y semanales bien definidos— se capturan con vocabularios reducidos (K = 3-4) cuya información discreta se complementa con el residual intra-régimen de la variante *soft residual*. El K-sweep análogo en modo --features M con seed única (Plan B) genera **32 *caches* HMM-M** —8 K × 4 *datasets*— mediante la versión baum_welch_batch (sección 4.2.3); el óptimo Plan B se reporta en la Tabla 6 (sección 4.7) y se discute en 6.5.

Para ilustrar el comportamiento del entrenamiento, la Ilustración 11 muestra la curva de convergencia sobre ETTh2. La log-verosimilitud es estrictamente creciente a lo largo de las iteraciones, validando que la implementación respeta la propiedad teórica del algoritmo EM (Dempster et al., 1977).

<span id="_Ref226882016" class="anchor"></span>Ilustración 11. Curva de convergencia de Baum-Welch sobre ETTh2 (K = 5): log-verosimilitud monótonamente creciente y \|ΔLL\| decreciente exponencialmente hasta el umbral ε = 10⁻⁴.

<img src="./media/image13.png" style="width:6.29861in;height:1.77014in" />Fuente: Elaboración propia.

El incremento \|ΔLL\| decrece exponencialmente, cruzando el umbral de convergencia de forma suave sin oscilaciones.

Los parámetros entrenados se almacenan en caché persistente para garantizar reproducibilidad exacta entre ejecuciones.

La validación de los parámetros estimados verifica propiedades estocásticas fundamentales: cada fila de A suma 1 y π suma 1. La diagonal de A presenta consistentemente valores elevados en todos los *datasets*, indicando que los regímenes son persistentes —el modelo permanece en el mismo estado durante múltiples *timesteps* antes de transitar a otro—. Las medias $\mu_{k}$ se distribuyen a lo largo del rango normalizado, cubriendo los distintos niveles de la serie, y las desviaciones $\sigma_{k}$ capturan regímenes de diferente volatilidad.

## Fase 3: Tokenización (Viterbi)

El algoritmo de Viterbi se implementa también en espacio logarítmico. Recibe la serie normalizada y los parámetros HMM entrenados, y devuelve la secuencia óptima de estados $Q^{*} = \left\lbrack z_{1},\ldots,z_{T} \right\rbrack$ junto con la log-probabilidad del camino óptimo.

La tokenización produce una asignación estado-*timestep* que puede comprimirse mediante *run-length encoding* —agrupando *timesteps* consecutivos con el mismo estado en un único segmento—. El ratio de compresión resultante depende del *dataset* y del K seleccionado, reflejando las diferencias en la estructura temporal de cada serie. A modo de ejemplo ilustrativo, sobre ETTh2 con K = 5 se obtiene un ratio de 4.79x sobre la partición de entrenamiento. Los valores agregados sobre 3 *seeds* en las configuraciones óptimas (Tabla 5) se reportan junto con la métrica de persistencia de *tokens* en la sección 5.6, dentro de la comparativa intrínseca frente a las demás técnicas. En todos los *datasets* evaluados, la distribución de *tokens* es balanceada: cada estado captura una proporción razonable de los *timesteps*, sin estados dominantes ni degenerados.

<span id="_Toc229829353" class="anchor"></span>Ilustración 12. Tokenización Viterbi sobre ETTh2 con K = 5 (ejemplo ilustrativo): asignación de estados, secuencia de tokens y distribución de frecuencias.

<img src="./media/image14.png" style="width:6.29861in;height:2.73958in" />Fuente: Elaboración propia.

La persistencia de los *tokens* HMM —longitud promedio de *runs* consecutivos del mismo estado— es consistentemente superior a la de la discretización, que asigna símbolos de forma puntual sin memoria temporal. Esta diferencia confirma que los *tokens* HMM representan segmentos temporales coherentes, no asignaciones aisladas.

## Variantes de tokenización HMM

Durante el desarrollo se identificó una limitación del enfoque *hard* (Viterbi): los *timesteps* asignados al mismo estado reciben un *embedding* idéntico, independientemente de su valor real. Dos observaciones con valores distintos que caigan en el mismo régimen resultan indistinguibles para el *transformer*. Este cuello de botella de información se manifiesta empíricamente como predicciones planas próximas a la media del régimen dominante.

Para resolver esta limitación se implementan las dos variantes descritas en la sección 4.2.7. La variante *soft* emplea las probabilidades *a posteriori* del algoritmo *Forward-Backward* para ponderar los *embeddings* base, produciendo un vector continuo y único por *timestep*. La variante *soft residual* añade además un término que codifica la posición de la observación dentro del régimen asignado. El *pipeline* soporta las dos variantes evaluadas en el Plan A como técnicas independientes; la variante *hard* (Viterbi *argmax*) queda excluida *a priori* (sección 4.2.7) por el cuello de botella de gradiente que introduce.

## Fase 4: Embeddings estructurados

La generación de *embeddings* se implementa como un módulo que, a partir de los parámetros HMM cargados desde caché, construye la tabla de *embeddings* crudos de dimensión $\lbrack K,2 + K\rbrack\ $según la definición de la sección 4.2.6, y una proyección lineal entrenable de dimensión $\left\lbrack 2 + K,d_{model} \right\rbrack\ $para compatibilidad con el *transformer*. Con $K\  = \ 5$, por ejemplo, cada *embedding* crudo tiene dimensión 7: dos componentes estadísticos $\left( \mu_{k},\sigma_{k} \right)\ $y cinco probabilidades de transición $(A\lbrack k,:\rbrack)$. La proyección a $d_{model}\ $es entrenable *end-to-end* junto con el *transformer* durante la tarea de predicción.

<span id="_Ref226882080" class="anchor"></span>Ilustración 13. Espacio de *embeddings* HMM sobre ETTh2 K = 5 (ejemplo ilustrativo): plano $\mu_{k} - \ \sigma_{k}$ de los cinco regímenes y matriz de transición A asociada.

<img src="./media/image15.png" style="width:6.29861in;height:2.21875in" />Fuente: Elaboración propia.

La Ilustración 13 visualiza el espacio de *embeddings* en dos paneles. El panel izquierdo proyecta los estados en el plano $\mu\  - \ \sigma$, donde el eje horizontal representa el centro del régimen y el vertical su volatilidad. Los estados se distribuyen a lo largo del rango normalizado con volatilidades diferenciadas, confirmando que el HMM identifica regímenes estadísticamente distintos — no variantes del mismo patrón. El panel derecho muestra la matriz de transición A. La diagonal presenta valores elevados, indicando regímenes persistentes. Las transiciones *off-diagonal* no son uniformes sino estructuradas: ciertos pares de estados presentan transiciones frecuentes mientras que otros son prácticamente inaccesibles, reflejando la dinámica real de la serie.

La interpretabilidad de estos *embeddings* es una propiedad diferencial frente a otras técnicas. En *patching*, cada *token* es un vector de P valores crudos cuyas dimensiones no admiten inspección directa. En *foundation models*, los *embeddings* son representaciones latentes de alta dimensionalidad sin correspondencia con magnitudes observables. En los *embeddings* HMM, cada componente tiene un significado concreto: $\mu_{k}\ $indica el nivel típico del régimen en unidades normalizadas, $\sigma_{k}$ cuantifica su volatilidad, y $A\lbrack k,k\rbrack\ $determina su persistencia media —un estado con $A\lbrack k,k\rbrack = 0.88\ $persiste en promedio $\Rightarrow \frac{1}{1 - 0.88} \approx 8$ *timesteps* antes de transitar—. Esta transparencia permite inspeccionar y validar cualitativamente la representación aprendida, lo que no es posible con los *embeddings* de caja negra de las demás técnicas.

## Comparativa intrínseca de técnicas de tokenización

Las diez métricas intrínsecas definidas en la sección 4.5.1 se implementan en un módulo de evaluación que recibe la tokenización de cualquiera de las seis técnicas y calcula las métricas aplicables. La comparativa se ejecuta sobre los cuatro *datasets* del Grupo 1 (ETTh1, ETTh2, Weather, Electricity), agregando los resultados como `mean ± std` sobre las 3 *seeds* {42, 2021, 7} descritas en la sección 4.7. Las técnicas determinísticas y el HMM en su K seleccionado producen `std ≈ 0` por construcción en las métricas intrínsecas —las técnicas no incorporan estocasticidad y el HMM converge a parámetros idénticos *seed* a *seed* cuando K coincide con el óptimo robusto—; MOMENT-inspired sí presenta `std` no nulo porque su muestreador de enmascarado depende de la *seed*.

**Métricas universales.** Las Tablas 8-11 reportan el ratio de compresión, el error de reconstrucción y la retención de autocorrelación para los cuatro *datasets* del Grupo 1.

<span id="_Ref228534824" class="anchor"></span>Tabla 8. Métricas intrínsecas universales — ETTh1, mean ± std sobre 3 seeds.

| Tokenizador         | Ratio de compresión | MSE de reconstrucción | Retención ACF   |
|---------------------|---------------------|-----------------------|-----------------|
| SAX-inspired        | 1.00 ± 0.00         | 0.1281 ± 0.0000       | 0.9997 ± 0.0000 |
| LLMTime-inspired    | 0.10 ± 0.00         | 4.00·10⁻⁵ ± 0         | 1.0000 ± 0.0000 |
| PatchTST-inspired   | 16.00 ± 0.00        | 6.34·10⁻¹⁶ ± 0        | 1.0000 ± 0.0000 |
| Autoformer-inspired | 4 320 ± 0           | 4.08·10⁻³⁴ ± 0        | 1.0000 ± 0.0000 |
| MOMENT-inspired     | 16.00 ± 0.00        | 0.3077 ± 0.0054       | 0.9990 ± 0.0003 |
| HMM (K=8 soft)      | 1.00 ± 0.00         | 0.0450 ± 0.0000       | 0.9998 ± 0.0000 |

Fuente: Elaboración propia.

<span id="_Toc229829381" class="anchor"></span>Tabla 9. Métricas intrínsecas universales — ETTh2, mean ± std sobre 3 seeds.

| Tokenizador         | Ratio de compresión | MSE de reconstrucción | Retención ACF   |
|---------------------|---------------------|-----------------------|-----------------|
| SAX-inspired        | 1.00 ± 0.00         | 0.1249 ± 0.0000       | 0.9998 ± 0.0000 |
| LLMTime-inspired    | 0.09 ± 0.00         | 4.16·10⁻⁵ ± 0         | 1.0000 ± 0.0000 |
| PatchTST-inspired   | 16.00 ± 0.00        | 6.53·10⁻¹⁶ ± 0        | 1.0000 ± 0.0000 |
| Autoformer-inspired | 4 320 ± 0           | 5.08·10⁻³⁴ ± 0        | 1.0000 ± 0.0000 |
| MOMENT-inspired     | 16.00 ± 0.00        | 0.3120 ± 0.0129       | 0.9975 ± 0.0002 |
| HMM (K=9 soft)      | 1.00 ± 0.00         | 0.0363 ± 0.0000       | 0.9999 ± 0.0000 |

Fuente: Elaboración propia.

<span id="_Toc229829382" class="anchor"></span>Tabla 10. Métricas intrínsecas universales — Weather, mean ± std sobre 3 seeds.

| Tokenizador         | Ratio de compresión | MSE de reconstrucción | Retención ACF   |
|---------------------|---------------------|-----------------------|-----------------|
| SAX-inspired        | 1.00 ± 0.00         | 0.1443 ± 0.0000       | 0.9999 ± 0.0000 |
| LLMTime-inspired    | 0.09 ± 0.00         | 4.10·10⁻⁵ ± 0         | 1.0000 ± 0.0000 |
| PatchTST-inspired   | 16.00 ± 0.00        | 6.42·10⁻¹⁶ ± 0        | 1.0000 ± 0.0000 |
| Autoformer-inspired | 18 432 ± 0          | 8.59·10⁻³⁵ ± 0        | 1.0000 ± 0.0000 |
| MOMENT-inspired     | 16.00 ± 0.00        | 0.2964 ± 0.0066       | 0.9964 ± 0.0002 |
| HMM (K=4 soft-res)  | 1.00 ± 0.00         | 0.1398 ± 0.0000       | 0.9989 ± 0.0000 |

Fuente: Elaboración propia.

<span id="_Toc229829383" class="anchor"></span>Tabla 11. Métricas intrínsecas universales — Electricity, mean ± std sobre 3 seeds.

| Tokenizador         | Ratio de compresión | MSE de reconstrucción | Retención ACF   |
|---------------------|---------------------|-----------------------|-----------------|
| SAX-inspired        | 1.00 ± 0.00         | 0.1648 ± 0.0000       | 0.9997 ± 0.0000 |
| LLMTime-inspired    | 0.10 ± 0.00         | 4.20·10⁻⁵ ± 0         | 1.0000 ± 0.0000 |
| PatchTST-inspired   | 16.00 ± 0.00        | 6.31·10⁻¹⁶ ± 0        | 1.0000 ± 0.0000 |
| Autoformer-inspired | 9 168 ± 0           | 5.32·10⁻³⁴ ± 0        | 1.0000 ± 0.0000 |
| MOMENT-inspired     | 16.00 ± 0.00        | 0.2993 ± 0.0036       | 0.9980 ± 0.0002 |
| HMM (K=3 soft-res)  | 1.00 ± 0.00         | 0.1577 ± 0.0000       | 0.9995 ± 0.0000 |

Fuente: Elaboración propia.

El **ratio de compresión** queda fijado por construcción: PatchTST-inspired y MOMENT-inspired comparten P = 16 (compresión 16×); Autoformer-inspired colapsa la entrada a dos componentes —*trend* y *seasonal*— produciendo el ratio más alto (4 320× en ETTh1/ETTh2, 18 432× en Weather, 9 168× en Electricity); SAX-inspired y HMM mantienen granularidad 1:1; LLMTime-inspired expande la secuencia ~10×.

El **error de reconstrucción** separa con claridad las técnicas *lossless* —Autoformer-inspired, PatchTST-inspired y LLMTime-inspired alcanzan precisión numérica— de las *lossy*. MOMENT-inspired presenta el mayor MSE (~0.30 en los cuatro *datasets*) por el enmascarado del 30 % de *patches*. Entre las *lossy*, **HMM reconstruye con menor MSE que SAX-inspired en los cuatro** *datasets* —reducciones del 64.9 % en ETTh1 (0.045 vs 0.128), 70.9 % en ETTh2 (0.036 vs 0.125), 3.1 % en Weather (0.140 vs 0.144) y 4.3 % en Electricity (0.158 vs 0.165)—. Es relevante que el HMM utilice vocabularios entre K = 3 y K = 9, comparables o inferiores a los 8 símbolos de SAX, y aun así capture mejor la magnitud original de la observación.

La **retención de autocorrelación** es ≥ 0.996 en todas las celdas: ninguna de las seis técnicas destruye la estructura de dependencias temporales de la serie; la métrica no discrimina sobre estos *datasets*.

**Métricas discretas (HMM vs SAX-inspired).** Las Tablas 12-15 comparan SAX-inspired y HMM —decodificado por Viterbi para que el conjunto de *tokens* sea finito y comparable— sobre las cuatro métricas discretas.

<span id="_Ref228534808" class="anchor"></span>Tabla 12. Métricas intrínsecas discretas — ETTh1, mean ± std sobre 3 seeds.

| Tokenizador    | Entropía vocab. | Entropía bigramas | Persistencia tokens | Cobertura top-5 |
|----------------|-----------------|-------------------|---------------------|-----------------|
| SAX-inspired   | 0.9989 ± 0.0000 | 0.8120 ± 0.0000   | 2.09 ± 0.00         | 0.6532 ± 0.0000 |
| HMM (K=8 soft) | 0.9795 ± 0.0000 | 0.7453 ± 0.0000   | 3.17 ± 0.00         | 0.7296 ± 0.0002 |

Fuente: Elaboración propia.

<span id="_Toc229829385" class="anchor"></span>Tabla 13. Métricas intrínsecas discretas — ETTh2, mean ± std sobre 3 seeds.

| Tokenizador    | Entropía vocab. | Entropía bigramas | Persistencia tokens | Cobertura top-5 |
|----------------|-----------------|-------------------|---------------------|-----------------|
| SAX-inspired   | 0.9892 ± 0.0000 | 0.7491 ± 0.0000   | 2.74 ± 0.00         | 0.6986 ± 0.0000 |
| HMM (K=9 soft) | 0.9868 ± 0.0000 | 0.7750 ± 0.0000   | 2.81 ± 0.00         | 0.6579 ± 0.0000 |

Fuente: Elaboración propia.

<span id="_Toc229829386" class="anchor"></span>Tabla 14. Métricas intrínsecas discretas — Weather, mean ± std sobre 3 seeds.

| Tokenizador        | Entropía vocab. | Entropía bigramas | Persistencia tokens | Cobertura top-5 |
|--------------------|-----------------|-------------------|---------------------|-----------------|
| SAX-inspired       | 0.9753 ± 0.0000 | 0.6718 ± 0.0000   | 4.12 ± 0.00         | 0.7325 ± 0.0000 |
| HMM (K=4 soft-res) | 0.9865 ± 0.0000 | 0.5796 ± 0.0000   | 18.24 ± 0.00        | 1.0000 ± 0.0000 |

Fuente: Elaboración propia.

<span id="_Toc229829387" class="anchor"></span>Tabla 15. Métricas intrínsecas discretas — Electricity, mean ± std sobre 3 seeds.

| Tokenizador        | Entropía vocab. | Entropía bigramas | Persistencia tokens | Cobertura top-5 |
|--------------------|-----------------|-------------------|---------------------|-----------------|
| SAX-inspired       | 0.9881 ± 0.0000 | 0.8001 ± 0.0000   | 1.93 ± 0.00         | 0.7128 ± 0.0000 |
| HMM (K=3 soft-res) | 0.9969 ± 0.0000 | 0.6887 ± 0.0001   | 7.74 ± 0.00         | 1.0000 ± 0.0000 |

Fuente: Elaboración propia.

La **entropía de vocabulario** está ≥ 0.97 en todas las celdas: ambas técnicas usan el vocabulario de forma equilibrada por construcción —SAX por *breakpoints* gaussianos equiprobables, HMM por inicialización *k-means* a K bajo—. La métrica no discrimina.

La **entropía de bigramas** sí discrimina. El HMM produce transiciones más predecibles que SAX en ETTh1 (Δ = −0.067), Weather (Δ = −0.092) y Electricity (Δ = −0.111), confirmando que la matriz de transición A impone estructura no trivial sobre los pares de *tokens* consecutivos. En ETTh2 la diferencia se invierte (Δ = +0.026), efecto colateral del K = 9 elegido: un vocabulario más rico infla el soporte de bigramas.

La **persistencia de** *tokens* —la métrica más informativa— muestra el mayor contraste: los *runs* del HMM son entre 1.02× y 4.42× más largos que los de SAX (1.52× ETTh1, 1.02× ETTh2, 4.42× Weather, 4.01× Electricity), con los mayores contrastes en Weather y Electricity. La amplitud cross-*dataset* del HMM (2.81 → 18.24) refleja la estructura real de regímenes en cada serie; la amplitud de SAX (1.93 → 4.12) refleja únicamente la suavidad de la señal de entrada.

La **cobertura top-5** satura en 1.0 para HMM en Weather (K = 4) y Electricity (K = 3): con $K\  \leq 5$ todos los *tokens* del vocabulario caben en el top-5 y la métrica deja de ser informativa. En ETTh1 y ETTh2 los valores reflejan el mismo patrón que la entropía de vocabulario.

**Robustez bajo perturbación.** La Tabla 16 resume el cambio relativo del MSE de reconstrucción al perturbar la entrada con ruido gaussiano de desviación $\sigma \in \text{\{}0.1 \cdot \sigma_{X},\, 0.5 \cdot \sigma_{X}\text{\}}$ (cross-*dataset*, mean ± std). La Tabla 17 desglosa el caso $\sigma = \ 0.5$ por *dataset* y la Tabla 18 reporta la métrica equivalente en el espacio discreto o continuo de *tokens*, según corresponda a cada técnica.

<span id="_Ref228532293" class="anchor"></span>Tabla 16. Robustez intrínseca — cambio relativo en MSE de reconstrucción bajo perturbación gaussiana, resumen cross-dataset.

| Tokenizador         | σ = 0.1·σₓ      | σ = 0.5·σₓ         |
|---------------------|-----------------|--------------------|
| SAX-inspired        | −0.3 % ± 2.2 %  | −26.3 % ± 8.5 %    |
| LLMTime-inspired    | +0.5 % ± 2.5 %  | +1.3 % ± 3.3 %     |
| PatchTST-inspired   | +6.0 % ± 13.1 % | +125.1 % ± 212.2 % |
| Autoformer-inspired | −5.8 % ± 6.7 %  | +21.3 % ± 36.9 %   |
| MOMENT-inspired     | +1.0 % ± 0.5 %  | +24.6 % ± 1.8 %    |
| HMM (RITMO)         | +5.4 % ± 2.2 %  | +126.4 % ± 66.2 %  |

Fuente: Elaboración propia.

<span id="_Ref228532301" class="anchor"></span>Tabla 17. Robustez intrínseca — cambio relativo en MSE de reconstrucción bajo σ = 0.5·σₓ por dataset.

| Tokenizador         | ETTh1   | ETTh2    | Weather    | Electricity |
|---------------------|---------|----------|------------|-------------|
| SAX-inspired        | −14.5 % | −25.2 %  | −28.6 %    | −36.9 %     |
| LLMTime-inspired    | +1.1 %  | −0.5 %   | +6.0 %     | −1.3 %      |
| PatchTST-inspired   | +27.8 % | +21.8 %  | +457.0 % † | −6.3 %      |
| Autoformer-inspired | −8.4 %  | +0.2 %   | +80.8 % †  | +12.5 %     |
| MOMENT-inspired     | +24.3 % | +25.2 %  | +24.6 %    | +24.2 %     |
| HMM (RITMO)         | +93.0 % | +233.6 % | +101.4 %   | +77.3 %     |

Fuente: Elaboración propia. En Weather, el MSE basal de las técnicas *lossless* (PatchTST-inspired, Autoformer-inspired) cae por debajo de 10⁻¹⁵; el porcentaje queda dominado por un denominador minúsculo y no es directamente comparable con HMM/SAX/MOMENT — véase la Tabla 18 para la medida equivalente en el espacio de *tokens*.

<span id="_Ref228532275" class="anchor"></span>Tabla 18. Distancia en el espacio de tokens entre serie original y perturbada bajo σ = 0.5·σₓ, mean sobre 3 seeds. Edit distance normalizada a \[0, 1\] para técnicas discretas; L₂ en unidades del espacio de tokens para técnicas continuas.

| Método                   | ETTh1   | ETTh2   | Weather | Electricity |
|--------------------------|---------|---------|---------|-------------|
| SAX-inspired (edit)      | 0.5249  | 0.5152  | 0.5277  | 0.5098      |
| HMM (RITMO, edit)        | 0.4921  | 0.4799  | 0.2259  | 0.1540      |
| PatchTST-inspired (L₂)   | 1.9550  | 1.9550  | 1.9710  | 1.9637      |
| Autoformer-inspired (L₂) | 12.2347 | 12.2347 | 25.3530 | 17.8126     |
| MOMENT-inspired (L₂)     | 1.9550  | 1.9550  | 1.9710  | 1.9637      |

Fuente: Elaboración propia.

A nivel agregado (Tabla 16), SAX-inspired es la técnica más estable —paradójicamente reduce el MSE bajo $\sigma = \ 0.5$ porque el ruido cruza los *breakpoints* gaussianos—, LLMTime-inspired es prácticamente invariante (la serialización por dígitos absorbe perturbaciones por debajo de la última cifra decimal), y MOMENT-inspired aumenta el MSE ~25 % de forma estable. **PatchTST-inspired y HMM son los más sensibles** en MSE de reconstrucción (+125 % y +126 % respectivamente bajo $\sigma = \ 0.5$). El máximo del HMM aparece en ETTh2 (+233.6 %), mismo *dataset* en que el HMM gana el MSE *downstream* a horizonte largo: el vocabulario K = 9 que absorbe el *distribution shift* train/test también amplifica la respuesta a perturbaciones intra-partición.

La Tabla 18 revela un patrón distinto: **la secuencia de** *tokens* **del HMM cambia menos que la de SAX bajo perturbación en los cuatro** *datasets*. La distancia de edición normalizada del HMM es 6.2 % menor en ETTh1, 6.9 % menor en ETTh2, 57.2 % menor en Weather y 69.8 % menor en Electricity. La matriz de transición A penaliza las transiciones rápidas y regulariza la posterior $\gamma_{t}(k)$, estabilizando la secuencia decodificada por Viterbi incluso cuando la reconstrucción continua se ve más afectada.

Esta dualidad —reconstrucción sensible, secuencia discreta estable— es una propiedad estructural del marco markoviano y no aparece en ninguna otra técnica evaluada.

El análisis cuantitativo de cómo estas métricas intrínsecas se relacionan con el desempeño *downstream* y la regla empírica que separa los escenarios donde la tokenización HMM es competitiva se desarrolla en el Capítulo 7.

## Fase 5: Modelo de predicción (Transformer)

La arquitectura del *TransformerCommon* y el flujo de integración con las seis técnicas de tokenización se han descrito en la sección 4.2.8 dentro del marco teórico. Esta sección documenta las decisiones de implementación tomadas durante el desarrollo, especialmente la búsqueda empírica de hiperparámetros y los intervalos explorados, que constituyen la parte más sustancial del trabajo de simulación realizado.

**Selección de hiperparámetros: barrido empírico.** La configuración final se obtiene mediante un proceso iterativo de búsqueda manual sobre ETTh1 y Weather, partiendo de configuraciones reducidas y escalando hasta encontrar el equilibrio óptimo entre capacidad y regularización para las seis técnicas. La Tabla 19 documenta los intervalos explorados para cada hiperparámetro, el valor seleccionado y el criterio que motivó la decisión. La búsqueda exploró del orden de cuarenta combinaciones distintas de $\left( d_{model},\ n_{heads},\ e_{layers},\ d_{ff},\ scheduler,\ epochs \right)$ antes de fijar la configuración definitiva, con el objetivo explícito de no favorecer a ninguna técnica en particular. Esta configuración se aplica idéntica en Plan A y Plan B; las únicas asimetrías entre planes —train_epochs por *dataset* (ETT 30, Weather 10, Electricity 5), batch_size 16 en Electricity y K-óptimo distinto por *dataset*— se documentan en la Tabla 7 (sección 5.1.2) y la Tabla 6 (sección 4.7), y responden a restricciones de presupuesto GPU.

<span id="_Ref226879947" class="anchor"></span>Tabla 19. Intervalos explorados, valor seleccionado y justificación de los hiperparámetros del *transformer* común.

| Parámetro      | Intervalo explorado                     | Valor final         | Justificación                                                                                                                  |
|----------------|-----------------------------------------|---------------------|--------------------------------------------------------------------------------------------------------------------------------|
| $$d_{model}$$  | {16, 32, **64**, 128, 256}              | 64                  | Modelos ≥128 sobreajustan con embeddings HMM (5–10); ≤32 penalizan patching. 64 maximiza el promedio.                          |
| $$n_{heads}$$  | {2, **4**, 8}                           | 4                   | $d_{head} = 16$. Con 8 cabezas ($d_{head} = 8$) baja expresividad; con 2 baja diversidad.                                      |
| $$e_{layers}$$ | {1, **2**, 3, 4}                        | 2                   | Más capas no mejoran con secuencias cortas (96 tokens) y aumentan sobreajuste.                                                 |
| $$d_{ff}$$     | {32, 64, **128**, 256, 512}             | 128                 | Ratio $2\  \times \ d_{model}$; $4\  \times \ ( = 256)\ $sobreajusta en Grupo 1.                                               |
| dropout        | {0.0, **0.1**, 0.2, 0.3}                | 0.1                 | ≥0.2 degrada MSE en técnicas continuas.                                                                                        |
| learning rate  | {10⁻⁴, 5·10⁻⁴, **10⁻³**, 5·10⁻³}        | 10⁻³                | Con cosine annealing; menor no converge, mayor diverge.                                                                        |
| scheduler      | {type1, constant, **cosine annealing**} | cosine annealing    | type1 reduce demasiado rápido; cosine mantiene aprendizaje y mejora $MSE(\sim 4\%)$.                                           |
| epochs         | {10, 20, **30**, 50}                    | 30                  | Con early stopping $(patience = 7)$; converge entre 15–25.                                                                     |
| batch size     | {16, **32**, 64, 128}                   | 32                  | Batches grandes empeoran MSE en datasets pequeños.                                                                             |
| K (HMM)        | {3, 4, 5, 6, 7, 8, 9, 10}               | Dependiente dataset | $K = 8\ $(ETTh1, *soft*), $K = 9$ (ETTh2, *soft*), $K = 4$ (Weather, *soft residual*), $K = 3$ (Electricity, *soft residual*). |
| Variante HMM   | {hard, soft, soft residual}             | Dependiente dataset | *soft* en ETTh1 y ETTh2; *soft residual* en Weather y Electricity.                                                             |
| Optimizador    | Adam                                    | Adam                | Inicialización Xavier uniform.                                                                                                 |
| Loss           | MSE tras RevIN⁻¹                        | —                   | Calculada en escala original.                                                                                                  |

Fuente: Elaboración propia.

La elección de *cosine annealing* frente al *scheduler type1* (reducción por factor cada *epoch*) se fundamenta en una observación empírica documentada en los apuntes de desarrollo del proyecto: el *scheduler type1* divide el *learning rate* por 2 en cada *epoch*, reduciendo la tasa a valores insignificantes antes de que el modelo haya convergido. Con *cosine annealing*, la tasa decrece gradualmente, manteniendo capacidad de aprendizaje durante más *epochs*. Análogamente, el barrido sobre $d_{model}$ reveló un efecto inesperado: aumentar la capacidad del *transformer* de $d_{model} = 16\ a\ d_{model} = 64\ $mejoró sistemáticamente el MSE de las técnicas continuas (*patching*, *foundation*), pero empeoró el de HMM (sobreajuste de los *embeddings* de baja dimensión); aumentar a $d_{model} = 128\ $sobreajustó todas las técnicas. $d_{model} = 64\ $representa el punto de equilibrio que no favorece a ninguna técnica.

# RESULTADOS

Este capítulo presenta los resultados experimentales obtenidos al ejecutar los dos escenarios definidos en la sección 5.1. La sección 6.1 documenta el protocolo de ejecución del Plan A. La sección 6.2 reporta los resultados del Plan A sobre los cuatro *datasets* del Grupo 1, organizados por *dataset* y horizonte. La sección 6.3 reporta la evaluación de transferencia *cross-domain* del *tokenizer* HMM sobre los dos *datasets* del Grupo 2. La sección 6.4 muestra la inspección cualitativa de los regímenes aprendidos. Finalmente, la sección 6.5 presenta los resultados del Plan B y la comparativa frente al estado del arte. La discusión y la interpretación detallada de estos resultados se desarrolla en el Capítulo 7.

## Protocolo de ejecución del Plan A

Los experimentos del Plan A ejecutan la comparación controlada de las seis técnicas de tokenización bajo la configuración descrita en la sección 5.7. Todas las técnicas comparten el mismo *TransformerCommon*, los mismos hiperparámetros, el mismo optimizador y el mismo *scheduler*, variando únicamente la etapa de tokenización y generación de *embeddings*. El Plan A comprende **504 ejecuciones controladas** en total: **288 ejecuciones** *in-domain* (6 técnicas × 4 horizontes × 4 *datasets* del Grupo 1 × 3 *seeds* {42, 2021, 7}) más **216 ejecuciones de transferencia** *cross-domain* **del** *tokenizer* **HMM** (9 fuentes de tokenizador × 4 horizontes × 2 *datasets* del Grupo 2 × 3 *seeds*). Cada cifra del Plan A reportada en las secciones 6.2-6.4 es el agregado mean ± std sobre las tres *seeds*; las cifras del Plan B (sección 6.5) se reportan como mediciones puntuales con seed única (2021) por restricción de presupuesto GPU.

Para cada combinación (*dataset*, horizonte, técnica) se ejecuta el ciclo completo de entrenamiento, validación y *test*. El *early stopping* detiene el entrenamiento cuando la pérdida de validación no mejora durante 7 *epochs* consecutivos, guardando el mejor modelo observado. La evaluación final se realiza sobre el conjunto de *test* con el modelo guardado, reportando MSE y MAE tras desnormalización RevIN.

Para las técnicas HMM, el número de estados K y la variante de *embedding* (*soft* / *soft residual*) se fijan por *dataset* según el óptimo robusto del barrido descrito en la sección 4.7 (Tabla 5): K = 8 *soft* en ETTh1, K = 9 *soft* en ETTh2, K = 4 *soft residual* en Weather y K = 3 *soft residual* en Electricity.

## Resultados Plan A: Grupo 1 (*in-domain*)

Los resultados completos por *dataset* y horizonte se presentan a continuación. Cada tabla incluye las seis técnicas ordenadas por MSE promedio sobre los cuatro horizontes O ∈ {96, 192, 336, 720}. Los valores en negrita indican el mejor resultado por horizonte; los valores en cursiva indican el segundo mejor.

<span id="_Ref228534763" class="anchor"></span>Tabla 20. Plan A — ETTh1: MSE por horizonte y MSE/MAE promedio sobre 3 seeds (mean ± std). Fila destacada en negrita: mejor avg MSE. Por columna, negrita marca el mejor valor y cursiva el segundo mejor.

| Técnica               | pl=96 MSE           | pl=192 MSE          | pl=336 MSE          | pl=720 MSE          | Avg MSE    | Avg MAE    |
|-----------------------|---------------------|---------------------|---------------------|---------------------|------------|------------|
| **PatchTST-inspired** | **0.0566 ± 0.0006** | **0.0751 ± 0.0008** | **0.0856 ± 0.0012** | *0.0934 ± 0.0015*   | **0.0777** | **0.2155** |
| MOMENT-inspired       | *0.0568 ± 0.0005*   | *0.0751 ± 0.0009*   | *0.0858 ± 0.0014*   | 0.0941 ± 0.0019     | *0.0779*   | *0.2158*   |
| Autoformer-inspired   | 0.0582 ± 0.0016     | 0.0779 ± 0.0009     | 0.0916 ± 0.0021     | **0.0857 ± 0.0041** | 0.0784     | 0.2178     |
| LLMTime-inspired      | 0.0597 ± 0.0003     | 0.0774 ± 0.0015     | 0.0932 ± 0.0014     | 0.0970 ± 0.0069     | 0.0818     | 0.2228     |
| RITMO K=8 soft        | 0.0596 ± 0.0018     | 0.0783 ± 0.0037     | 0.0925 ± 0.0011     | 0.0994 ± 0.0032     | 0.0824     | 0.2242     |
| SAX-inspired          | 0.0602 ± 0.0003     | 0.0783 ± 0.0017     | 0.0913 ± 0.0020     | 0.1016 ± 0.0032     | 0.0828     | 0.2248     |

Fuente: Elaboración propia.

**En ETTh1, RITMO ocupa la quinta posición** en avg MSE (0.0824), por detrás de PatchTST-inspired (0.0777, 1º), MOMENT-inspired (0.0779, 2º), Autoformer-inspired (0.0784, 3º) y LLMTime-inspired (0.0818, 4º), y solo por delante de SAX-inspired (0.0828, 6º). Las seis técnicas se concentran en una franja de un 6.5 % de amplitud (0.0777-0.0828). Por horizonte, PatchTST-inspired y MOMENT-inspired lideran pl = 96 y pl = 192, mientras que Autoformer-inspired toma la delantera en pl = 720 por un 8.2 % en MSE (0.0857 frente a 0.0934). El std *cross-seed* se mantiene por debajo de 0.0041 en las cinco mejores técnicas.

<span id="_Toc229829393" class="anchor"></span>Tabla 21. Plan A — ETTh2: MSE por horizonte y MSE/MAE promedio sobre 3 seeds (mean ± std). Fila destacada en negrita: mejor avg MSE. Por columna, negrita marca el mejor valor y cursiva el segundo mejor.

| Técnica             | pl=96 MSE           | pl=192 MSE          | pl=336 MSE          | pl=720 MSE          | Avg MSE    | Avg MAE    |
|---------------------|---------------------|---------------------|---------------------|---------------------|------------|------------|
| **RITMO K=9 soft**  | 0.1455 ± 0.0086     | 0.2004 ± 0.0131     | 0.2309 ± 0.0032     | **0.2294 ± 0.0066** | **0.2015** | *0.3548*   |
| PatchTST-inspired   | *0.1449 ± 0.0057*   | 0.2006 ± 0.0062     | 0.2225 ± 0.0103     | *0.2383 ± 0.0240*   | *0.2016*   | 0.3550     |
| Autoformer-inspired | 0.1490 ± 0.0092     | *0.1977 ± 0.0046*   | **0.2222 ± 0.0079** | 0.2385 ± 0.0020     | 0.2019     | **0.3541** |
| MOMENT-inspired     | 0.1456 ± 0.0057     | 0.2018 ± 0.0074     | *0.2224 ± 0.0096*   | 0.2397 ± 0.0254     | 0.2024     | 0.3558     |
| LLMTime-inspired    | **0.1444 ± 0.0049** | **0.1972 ± 0.0119** | 0.2279 ± 0.0035     | 0.2410 ± 0.0159     | 0.2026     | 0.3561     |
| SAX-inspired        | 0.1550 ± 0.0015     | 0.1992 ± 0.0045     | 0.2239 ± 0.0040     | 0.2661 ± 0.0328     | 0.2110     | 0.3627     |

Fuente: Elaboración propia.

**En ETTh2, RITMO con K = 9 *soft* obtiene el primer puesto en avg MSE** (0.2015), por un margen mínimo sobre PatchTST-inspired (0.2016, 2º) y Autoformer-inspired (0.2019, 3º, que se lleva el avg MAE: 0.3541). El comportamiento por horizonte es no-monótono: RITMO queda 3º en pl = 96 (LLMTime-inspired gana con 0.1444), 4º en pl = 192 (LLMTime-inspired vuelve a ganar con 0.1972), último en pl = 336 (Autoformer-inspired gana con 0.2222), y **mejor en pl = 720** (0.2294, un 3.7 % por debajo de PatchTST-inspired que queda 2º). SAX-inspired queda 6º (0.2110, +4.7 % sobre RITMO). ETTh2 es además el *dataset* donde la *seed* importa más: el std *cross-seed* alcanza 0.0240 para PatchTST-inspired en pl = 720, mientras que el de RITMO en el mismo horizonte (0.0066) es uno de los más bajos de la columna.

<span id="_Toc229829394" class="anchor"></span>Tabla 22. Plan A — Weather: MSE por horizonte y MSE/MAE promedio sobre 3 seeds (mean ± std). Fila destacada en negrita: mejor avg MSE. Por columna, negrita marca el mejor valor y cursiva el segundo mejor.

| Técnica                     | pl=96 MSE             | pl=192 MSE            | pl=336 MSE            | pl=720 MSE            | Avg MSE     | Avg MAE    |
|-----------------------------|-----------------------|-----------------------|-----------------------|-----------------------|-------------|------------|
| **RITMO K=4 soft residual** | **0.00124 ± 0.00003** | **0.00149 ± 0.00007** | 0.00166 ± 0.00004     | 0.00214 ± 0.00009     | **0.00163** | **0.0299** |
| LLMTime-inspired            | *0.00129 ± 0.00003*   | *0.00153 ± 0.00003*   | 0.00167 ± 0.00001     | *0.00212 ± 0.00003*   | *0.00165*   | 0.0303     |
| PatchTST-inspired           | 0.00133 ± 0.00004     | 0.00157 ± 0.00004     | **0.00162 ± 0.00000** | **0.00210 ± 0.00011** | *0.00165*   | *0.0301*   |
| Autoformer-inspired         | 0.00129 ± 0.00002     | 0.00156 ± 0.00006     | 0.00166 ± 0.00009     | 0.00216 ± 0.00012     | 0.00167     | 0.0304     |
| MOMENT-inspired             | 0.00130 ± 0.00003     | 0.00159 ± 0.00004     | *0.00165 ± 0.00004*   | 0.00214 ± 0.00005     | 0.00167     | 0.0303     |
| SAX-inspired                | 0.00150 ± 0.00006     | 0.00176 ± 0.00012     | 0.00191 ± 0.00018     | 0.00225 ± 0.00016     | 0.00185     | 0.0316     |

Fuente: Elaboración propia.

**En Weather, RITMO con K = 4 *soft residual* obtiene el primer puesto** en avg MSE (0.00163) y avg MAE (0.0299), gana pl = 96 con un 6.8 % sobre PatchTST-inspired y un 17.3 % sobre SAX-inspired, y gana pl = 192 con un 5.1 % y un 15.3 % respectivamente. A partir de pl = 336 las técnicas líderes se agrupan en una franja de menos del 4 % de amplitud y PatchTST-inspired toma la delantera en pl = 336 y pl = 720 por un margen reducido (0.00162 frente a 0.00166 en pl = 336), pero la ventaja temprana de RITMO en pl = 96 y pl = 192 arrastra el promedio. El std *cross-seed* se mantiene en 10⁻⁴ o inferior en los cuatro horizontes.

<span id="_Toc229829395" class="anchor"></span>Tabla 23. Plan A — Electricity: MSE por horizonte y MSE/MAE promedio sobre 3 seeds (mean ± std). Fila destacada en negrita: mejor avg MSE. Por columna, negrita marca el mejor valor y cursiva el segundo mejor.

| Técnica                 | pl=96 MSE           | pl=192 MSE          | pl=336 MSE          | pl=720 MSE          | Avg MSE    | Avg MAE    |
|-------------------------|---------------------|---------------------|---------------------|---------------------|------------|------------|
| **Autoformer-inspired** | *0.3026 ± 0.0065*   | **0.3233 ± 0.0080** | 0.3888 ± 0.0321     | **0.4142 ± 0.0113** | **0.3572** | **0.4304** |
| SAX-inspired            | 0.3149 ± 0.0043     | 0.3324 ± 0.0097     | **0.3797 ± 0.0061** | *0.4398 ± 0.0092*   | *0.3667*   | *0.4361*   |
| RITMO K=3 soft residual | 0.3100 ± 0.0062     | 0.3322 ± 0.0085     | *0.3809 ± 0.0109*   | 0.4533 ± 0.0136     | 0.3691     | 0.4361     |
| PatchTST-inspired       | **0.3020 ± 0.0053** | *0.3301 ± 0.0076*   | 0.3862 ± 0.0051     | 0.4765 ± 0.0132     | 0.3737     | 0.4376     |
| MOMENT-inspired         | 0.3078 ± 0.0117     | 0.3329 ± 0.0097     | 0.3879 ± 0.0060     | 0.4795 ± 0.0137     | 0.3770     | 0.4396     |
| LLMTime-inspired        | 0.3309 ± 0.0089     | 0.3538 ± 0.0055     | 0.4136 ± 0.0082     | 0.4942 ± 0.0281     | 0.3981     | 0.4561     |

Fuente: Elaboración propia.

**En Electricity, RITMO ocupa la tercera posición** en avg MSE (0.3691), por detrás de Autoformer-inspired (0.3572, 1º) y SAX-inspired (0.3667, 2º), y empatado en avg MAE con SAX-inspired (0.4361). PatchTST-inspired gana pl = 96 (0.3020) pero degrada con el horizonte hasta caer al 4º puesto en avg MSE. La comparativa visual por técnica sobre una misma muestra del conjunto de test de Electricity pl=96 se reporta en el Anexo D.

<span id="_Toc229829355" class="anchor"></span>Ilustración 14. Plan A — Curvas MSE frente a horizonte de predicción $O\  \in \text{\{}96,\ 192,\ 336,\ 720\text{\}}$ para los cuatro datasets del Grupo 1 y las seis técnicas comparadas, mean ± std sobre 3 seeds (288 ejecuciones in-domain).

<img src="./media/image16.png" style="width:6.29861in;height:4.32431in" />Fuente: Elaboración propia.

La comparativa global revela que **no existe una técnica dominante en los cuatro escenarios**. RITMO obtiene el primer puesto en avg MSE en Weather y ETTh2, queda en tercer puesto en Electricity y en quinto puesto en ETTh1. En tres de los cuatro *datasets* RITMO supera a SAX-inspired en avg MSE.

Para complementar las diferencias por columna observadas en las Tablas 20-23, la Tabla 24 reporta el test pareado de Wilcoxon de signo-rango entre RITMO y cada uno de los cinco *baselines* deterministas, sobre las 12 observaciones por *dataset* (4 horizontes × 3 *seeds*) con corrección de Bonferroni × 5 al nivel α = 0.01. Cada celda recoge tres cantidades: (i) el número de pares (horizonte, *seed*) en los que RITMO obtiene menor MSE que el *baseline* (*win count*), (ii) el *p*-valor corregido y (iii) la correlación rangos-pareados r como medida del tamaño del efecto (positiva: RITMO mejor; negativa: *baseline* mejor; magnitud en \[0, 1\]). Las celdas con **†** alcanzan significancia formal al nivel α = 0.01.

<span id="_Ref228534756" class="anchor"></span>Tabla 24. Plan A — Test pareado de Wilcoxon de signo-rango entre RITMO y los cinco baselines deterministas (n = 12 pares por celda; corrección Bonferroni × 5; α = 0.01).

| Dataset     | SAX-inspired                   | LLMTime-inspired               | PatchTST-inspired             | Autoformer-inspired     | MOMENT-inspired               |
|-------------|--------------------------------|--------------------------------|-------------------------------|-------------------------|-------------------------------|
| ETTh1       | 7/12, p=1.00, r = +0.26        | 6/12, p=1.00, r = −0.10        | **1/12, p=0.005†, r = −0.97** | 4/12, p=0.39, r = −0.59 | **1/12, p=0.005†, r = −0.97** |
| ETTh2       | 7/12, p=1.00, r = +0.33        | 6/12, p=1.00, r = +0.03        | 5/12, p=1.00, r = −0.05       | 7/12, p=1.00, r = ±0.00 | 4/12, p=1.00, r = −0.08       |
| Weather     | **11/12, p=0.005†, r = +0.97** | 7/12, p=1.00, r = +0.33        | 8/12, p=1.00, r = +0.33       | 9/12, p=0.39, r = +0.59 | 8/12, p=0.46, r = +0.56       |
| Electricity | 7/12, p=1.00, r = −0.15        | **11/12, p=0.005†, r = +0.97** | 7/12, p=1.00, r = +0.21       | 1/12, p=0.17, r = −0.69 | 9/12, p=0.65, r = +0.51       |

Fuente: Elaboración propia.

La Tabla 24 confirma estadísticamente el cuadro cualitativo de las Tablas 20-23. En **Weather**, RITMO es significativamente mejor que SAX-inspired (p = 0.005, r = +0.97); frente a las otras cuatro técnicas el *win count* favorece a RITMO (7/12-9/12) pero las diferencias absolutas son demasiado pequeñas para alcanzar significancia formal con n = 12 pares y corrección de Bonferroni × 5. En **Electricity**, RITMO es significativamente mejor que LLMTime-inspired (p = 0.005, r = +0.97), indistinguible de SAX-inspired/PatchTST-inspired/MOMENT-inspired y confirmado inferior a Autoformer-inspired (1/12 victorias). En **ETTh2**, la victoria en avg MSE no alcanza significancia formal: las victorias de RITMO se concentran en pl = 720 y 12 observaciones pareadas tras Bonferroni × 5 son insuficientes para discriminar. En **ETTh1**, RITMO es significativamente peor que PatchTST-inspired y MOMENT-inspired (p = 0.005, r = −0.97 en ambos casos) e indistinguible de SAX-inspired, LLMTime-inspired y Autoformer-inspired. En términos de tamaño del efecto, RITMO presenta dos ventajas formales (Weather frente a SAX-inspired; Electricity frente a LLMTime-inspired) y dos desventajas formales (ETTh1 frente a PatchTST-inspired y MOMENT-inspired).

<span id="_Toc229829356" class="anchor"></span>Ilustración 15. Plan A — Ranking heatmap de las seis técnicas por avg MSE en cada dataset del Grupo 1 (1 = mejor, 6 = peor; agregado sobre 3 seeds).

<img src="./media/image17.png" style="width:6.29861in;height:3.95764in" />Fuente: Elaboración propia.

## Resultados Plan A: Grupo 2 (transferencia *cross-domain* del *tokenizer* HMM)

Los *datasets* del Grupo 2 (Traffic y Exchange) se evalúan en modo de transferencia *cross-domain* del *tokenizer* HMM congelado, según la definición precisa introducida en la sección 5.1.1: el HMM se entrena sobre un *dataset* fuente del Grupo 1 y se reutiliza sin re-entrenar sus parámetros sobre el *dataset* objetivo, mientras que el *transformer* sí se entrena con los datos del *dataset* objetivo. Por brevedad, en las tablas se emplea el rótulo *zero-shot* como abreviación del término técnico anterior. Se evalúan las cuatro fuentes HMM correspondientes a las configuraciones óptimas del Grupo 1 (ETTh1 K=8 *soft*, ETTh2 K=9 *soft*, Weather K=4 *soft residual*, Electricity K=3 *soft residual*) y se reportan junto con las cinco técnicas determinísticas como referencia. Cada celda agrega mean ± std sobre 3 *seeds* {42, 2021, 7}; las técnicas determinísticas se reentrenan en el *dataset* objetivo, las cuatro fuentes HMM solo reentrenan el *transformer*. Total: 9 fuentes × 4 horizontes × 2 *datasets* × 3 *seeds* = **216 ejecuciones controladas**.

<span id="_Ref228535180" class="anchor"></span>Tabla 25. Plan A — Traffic (transferencia cross-domain del tokenizer HMM): MSE por horizonte y MSE/MAE promedio sobre 3 seeds (mean ± std). Fila destacada en negrita: mejor avg MSE. Por columna, negrita marca el mejor valor y cursiva el segundo mejor.

| Técnica                                      | pl=96 MSE           | pl=192 MSE          | pl=336 MSE          | pl=720 MSE          | Avg MSE    | Avg MAE    |
|----------------------------------------------|---------------------|---------------------|---------------------|---------------------|------------|------------|
| **MOMENT-inspired**                          | **0.1737 ± 0.0025** | **0.1667 ± 0.0027** | **0.1599 ± 0.0023** | **0.1796 ± 0.0024** | **0.1700** | **0.2544** |
| PatchTST-inspired                            | *0.1741 ± 0.0026*   | *0.1669 ± 0.0026*   | *0.1600 ± 0.0023*   | *0.1797 ± 0.0024*   | *0.1702*   | *0.2547*   |
| Autoformer-inspired                          | 0.1772 ± 0.0017     | 0.1705 ± 0.0063     | 0.1662 ± 0.0019     | 0.1887 ± 0.0004     | 0.1756     | 0.2600     |
| SAX-inspired                                 | 0.1982 ± 0.0057     | 0.1872 ± 0.0036     | 0.1791 ± 0.0034     | 0.1966 ± 0.0029     | 0.1903     | 0.2772     |
| RITMO K=4 soft residual (fuente Weather)     | 0.2110 ± 0.0021     | 0.2075 ± 0.0002     | 0.1893 ± 0.0117     | 0.2259 ± 0.0212     | 0.2084     | 0.2996     |
| RITMO K=9 soft (fuente ETTh2)                | 0.2689 ± 0.0890     | 0.2095 ± 0.0112     | 0.1998 ± 0.0124     | 0.2241 ± 0.0157     | 0.2256     | 0.3129     |
| RITMO K=3 soft residual (fuente Electricity) | 0.2393 ± 0.0187     | 0.2331 ± 0.0294     | 0.2071 ± 0.0066     | 0.2268 ± 0.0070     | 0.2266     | 0.3141     |
| LLMTime-inspired                             | 0.2702 ± 0.0110     | 0.2375 ± 0.0081     | 0.2341 ± 0.0093     | 0.2585 ± 0.0040     | 0.2501     | 0.3482     |
| RITMO K=8 soft (fuente ETTh1)                | 0.2952 ± 0.0937     | 0.2496 ± 0.0447     | 0.2412 ± 0.0352     | 0.2417 ± 0.0097     | 0.2569     | 0.3340     |

Fuente: Elaboración propia.

<span id="_Ref228535187" class="anchor"></span>Tabla 26. Plan A — Exchange (transferencia cross-domain del tokenizer HMM): MSE por horizonte y MSE/MAE promedio sobre 3 seeds (mean ± std). Fila destacada en negrita: mejor avg MSE. Por columna, negrita marca el mejor valor y cursiva el segundo mejor.

| Técnica                                      | pl=96 MSE           | pl=192 MSE          | pl=336 MSE          | pl=720 MSE          | Avg MSE    | Avg MAE    |
|----------------------------------------------|---------------------|---------------------|---------------------|---------------------|------------|------------|
| RITMO K=8 soft (fuente ETTh1)                | 0.1054 ± 0.0080     | 0.2189 ± 0.0052     | *0.4362 ± 0.0193*   | 1.0892 ± 0.0909     | **0.4624** | **0.4708** |
| PatchTST-inspired                            | **0.0987 ± 0.0039** | 0.2214 ± 0.0174     | 0.5106 ± 0.0482     | **1.0507 ± 0.1559** | *0.4704*   | *0.4754*   |
| MOMENT-inspired                              | *0.0990 ± 0.0044*   | 0.2213 ± 0.0171     | 0.5076 ± 0.0458     | *1.0613 ± 0.1678*   | 0.4723     | 0.4758     |
| RITMO K=4 soft residual (fuente Weather)     | 0.1271 ± 0.0071     | 0.2186 ± 0.0204     | 0.4949 ± 0.0510     | 1.2841 ± 0.1271     | 0.5312     | 0.5048     |
| LLMTime-inspired                             | 0.1154 ± 0.0044     | 0.2261 ± 0.0118     | 0.5092 ± 0.0635     | 1.2790 ± 0.1085     | 0.5324     | 0.5075     |
| RITMO K=3 soft residual (fuente Electricity) | 0.1110 ± 0.0056     | 0.2277 ± 0.0114     | 0.5351 ± 0.0216     | 1.2612 ± 0.0804     | 0.5337     | 0.5015     |
| Autoformer-inspired                          | 0.1226 ± 0.0260     | **0.2142 ± 0.0031** | 0.4512 ± 0.0440     | 1.3521 ± 0.0412     | 0.5350     | 0.4949     |
| RITMO K=9 soft (fuente ETTh2)                | 0.1205 ± 0.0054     | *0.2252 ± 0.0086*   | **0.4029 ± 0.0314** | 1.4092 ± 0.1411     | 0.5395     | 0.5025     |
| SAX-inspired                                 | 0.1177 ± 0.0091     | 0.2691 ± 0.0210     | 0.5619 ± 0.0201     | 1.4743 ± 0.0163     | 0.6057     | 0.5309     |

Fuente: Elaboración propia.

**En Traffic, los *baselines* deterministas reentrenados dominan**: MOMENT-inspired y PatchTST-inspired lideran los cuatro horizontes con avg MSE 0.1700 y 0.1702 respectivamente, mientras que la mejor fuente HMM (Weather, K = 4 *soft residual*) queda en quinta posición con avg MSE 0.2084 (un 22.6 % por encima del mejor *baseline*). La peor fuente HMM (ETTh1, K = 8 *soft*) cae al noveno puesto con avg MSE 0.2569 (un 51.1 % de degradación).

**En Exchange, el patrón se invierte**: el HMM congelado con fuente ETTh1 (K = 8 *soft*) **gana tanto avg MSE (0.4624) como avg MAE (0.4708)** sobre las cinco técnicas determinísticas reentrenadas. Ningún tokenizador domina los cuatro horizontes individuales —PatchTST-inspired gana pl = 96 y pl = 720 por menos del 7 %, Autoformer-inspired gana pl = 192, RITMO con fuente ETTh2 gana pl = 336—, pero la fila *RITMO K = 8 soft (fuente ETTh1)* es la única competitiva en los cuatro horizontes simultáneamente. Es además notable que la fuente HMM con mejor desempeño *in-domain* (Weather) no sea la que mejor transfiere a Exchange.

## Interpretabilidad de los regímenes aprendidos

Los regímenes aprendidos por el HMM admiten inspección directa de sus parámetros. La Ilustración 16 muestra la asignación de estados sobre un segmento de Weather con la configuración óptima del Plan A (K = 4 *soft residual*), donde cada color representa un régimen distinto. Los cuatro regímenes se corresponden con niveles de la serie normalizada: el estado con μₖ más bajo captura los valles, el estado con μₖ más alto captura los picos, y los dos estados intermedios cubren los regímenes de transición. La persistencia de cada estado se cuantifica con *runs* medios de ~18 *timesteps* sobre Weather, frente a ~4 de la discretización SAX-inspired (sección 5.6).

<span id="_Ref228535802" class="anchor"></span>Ilustración 16. Tokenización HMM sobre Weather (K = 4 soft residual, configuración óptima del Plan A): asignación de estados, secuencia de tokens y distribución de frecuencias.

<img src="./media/image18.png" style="width:6.29861in;height:2.73958in" />Fuente: Elaboración propia.

<span id="_Ref228558221" class="anchor"></span>Ilustración 17. HMM Weather (K = 4 soft residual) — izquierda: espacio de embeddings μₖ−σₖ con callouts por estado (μₖ, σₖ, fₖ, A\[k, k\]); derecha: matriz de transición A (media diagonal = 0.94).

<img src="./media/image19.png" style="width:6.29861in;height:2.6625in" />Fuente: Elaboración propia.

El **panel izquierdo** detalla los parámetros estadísticos por estado: S0 (μ = −1.22, σ = 0.27, f = 18.9 %, A\[0,0\] = 0.96) captura el régimen frío persistente; S2 (μ = −0.67, σ = 0.16, f = 24.6 %, A\[2,2\] = 0.93) es un régimen estable de baja volatilidad y alta frecuencia; S1 (μ = +0.04, σ = 0.28, f = 24.0 %, A\[1,1\] = 0.91) cubre el régimen central; y S3 (μ = +1.20, σ = 0.57, f = 32.5 %, A\[3,3\] = 0.96) corresponde al régimen extremo de alta volatilidad y máxima ocupación. La asignación cubre el rango normalizado de Weather (de μₖ ≈ −1.2 a μₖ ≈ +1.2) sin solapamientos, confirmando que los cuatro estados representan regímenes estadísticamente distintos. El **panel derecho** muestra la matriz de transición correspondiente con valores de auto-transición entre 0.91 y 0.96 (media 0.94), confirmando que los cuatro regímenes son altamente persistentes. Las transiciones *off-diagonal* no son uniformes: se concentran entre regímenes adyacentes en nivel (k y k ± 1), reflejando una estructura ordenada por nivel de la serie, mientras que las transiciones a regímenes no adyacentes son prácticamente inaccesibles.

## Resultados Plan B: comparación frente al estado del arte

Los experimentos del Plan B confrontan a RITMO-M con los cuatro *baselines* consolidados del estado del arte — DLinear (Zeng et al., 2023), PatchTST (Nie et al., 2023), TimeMixer (S. Wang et al., 2024), TimeXer (Y. Wang et al., 2024)— bajo la configuración descrita en la sección 5.1.2 (Tabla 7). Los cuatro *baselines* se ejecutan con los hiperparámetros publicados en sus *scripts* originales de TSLib, mientras que RITMO-M reutiliza el TransformerCommon del Plan A en modo --features M con la configuración óptima por *dataset* recogida en la Tabla 6 (sección 4.7, Ilustración 24 del Anexo C). El Plan B comprende **320 ejecuciones controladas** en total: **256 ejecuciones del K-sweep RITMO-M** (4 *datasets* × 2 variantes × 8 K × 4 horizontes) más **64 ejecuciones de los cuatro *baselines*** (4 modelos × 4 horizontes × 4 *datasets*). Por restricción de presupuesto GPU la evaluación se ejecuta sobre infraestructura RunPod RTX 3090 Ti con seed única (2021), sin transferencia *cross-domain* —ese escenario solo se reporta dentro del Plan A (sección 6.3)—; las cifras son por tanto mediciones puntuales y no admiten test estadístico formal (sección 4.5).

Los resultados completos por *dataset* y horizonte se presentan a continuación. Cada tabla incluye los cinco modelos ordenados por avg MSE sobre los cuatro horizontes O ∈ {96, 192, 336, 720}. Por columna, **negrita** marca el mejor valor y *cursiva* el segundo mejor; la fila destacada en negrita indica el ganador en avg MSE.

<span id="_Ref228557371" class="anchor"></span>Tabla 27. Plan B — ETTh1: MSE por horizonte y MSE/MAE promedio (seed única = 2021, `--features M`). Por columna, negrita marca el mejor valor y cursiva el segundo mejor.

| Modelo                  | pl=96 MSE  | pl=192 MSE | pl=336 MSE | pl=720 MSE | Avg MSE    | Avg MAE    |
|-------------------------|------------|------------|------------|------------|------------|------------|
| **TimeXer**             | 0.3818     | *0.4285*   | **0.4672** | 0.4988     | **0.4441** | 0.4451     |
| PatchTST                | **0.3774** | **0.4239** | *0.4676*   | 0.5259     | *0.4487*   | 0.4480     |
| TimeMixer               | *0.3778*   | 0.4397     | 0.5005     | **0.4785** | 0.4491     | **0.4400** |
| RITMO-M (hmm_soft, K=5) | 0.4006     | 0.4354     | 0.4744     | *0.4922*   | 0.4506     | *0.4439*   |
| DLinear                 | 0.3962     | 0.4450     | 0.4874     | 0.5126     | 0.4603     | 0.4568     |

Fuente: Elaboración propia.

**En ETTh1, RITMO-M ocupa la cuarta posición** en avg MSE (0.4506), por detrás de TimeXer (0.4441, 1º), PatchTST (0.4487, 2º) y TimeMixer (0.4491, 3º), y por delante de DLinear (0.4603, 5º). Las cinco columnas se concentran en una franja del 3.6 % de amplitud (0.4441-0.4603). Por horizonte: PatchTST gana pl = 96 (0.3774) y pl = 192 (0.4239), TimeXer gana pl = 336 (0.4672) y TimeMixer gana pl = 720 (0.4785). RITMO-M no es 1º en ningún horizonte individual pero **supera a DLinear en avg MSE en un 2.1 %** (0.4506 vs 0.4603), y obtiene la segunda posición en avg MAE (0.4439, *cursiva*).

<span id="_Ref228558297" class="anchor"></span>Tabla 28. Plan B — ETTh2: MSE por horizonte y MSE/MAE promedio (seed única = 2021, `--features M`). Por columna, negrita marca el mejor valor y cursiva el segundo mejor.

| Modelo                  | pl=96 MSE  | pl=192 MSE | pl=336 MSE | pl=720 MSE | Avg MSE    | Avg MAE    |
|-------------------------|------------|------------|------------|------------|------------|------------|
| **TimeXer**             | **0.2854** | **0.3628** | **0.4064** | 0.4485     | **0.3757** | **0.4005** |
| PatchTST                | 0.2921     | 0.3837     | *0.4195*   | *0.4385*   | *0.3834*   | 0.4115     |
| TimeMixer               | *0.2881*   | *0.3783*   | 0.4312     | 0.4579     | 0.3889     | *0.4083*   |
| RITMO-M (hmm_soft, K=3) | 0.3107     | 0.3961     | 0.4421     | 0.4380     | 0.3967     | 0.4141     |
| DLinear                 | 0.3414     | 0.4819     | 0.5929     | 0.8403     | 0.5642     | 0.5194     |

Fuente: Elaboración propia.

**En ETTh2, RITMO-M ocupa la cuarta posición** en avg MSE (0.3967), por detrás de TimeXer (0.3757, 1º), PatchTST (0.3834, 2º) y TimeMixer (0.3889, 3º), y supera a DLinear con un margen amplio (0.5642, 5º; −29.7 %). El comportamiento por horizonte es no-monótono: TimeXer domina los tres horizontes cortos y medios, pero **en pl = 720 RITMO-M gana a los cuatro *baselines***, con MSE de 0.4380 frente a 0.4385 (PatchTST), 0.4485 (TimeXer), 0.4579 (TimeMixer) y 0.8403 (DLinear).

<span id="_Ref228558282" class="anchor"></span>Tabla 29. Plan B — Weather: MSE por horizonte y MSE/MAE promedio (seed única = 2021, `--features M`). Por columna, negrita marca el mejor valor y cursiva el segundo mejor.

| Modelo                           | pl=96 MSE  | pl=192 MSE | pl=336 MSE | pl=720 MSE | Avg MSE    | Avg MAE    |
|----------------------------------|------------|------------|------------|------------|------------|------------|
| **TimeXer**                      | **0.1574** | **0.2041** | **0.2603** | **0.3399** | **0.2404** | **0.2705** |
| TimeMixer                        | *0.1612*   | *0.2072*   | *0.2628*   | 0.3463     | *0.2444*   | *0.2746*   |
| RITMO-M (hmm_soft_residual, K=5) | 0.1744     | 0.2215     | 0.2802     | 0.3534     | 0.2574     | 0.2804     |
| PatchTST                         | 0.1781     | 0.2214     | 0.2803     | 0.3557     | 0.2589     | 0.2810     |
| DLinear                          | 0.1962     | 0.2389     | 0.2811     | *0.3454*   | 0.2654     | 0.3169     |

Fuente: Elaboración propia.

**En Weather, RITMO-M ocupa la tercera posición** en avg MSE (0.2574), por detrás de TimeXer (0.2404, 1º) y TimeMixer (0.2444, 2º), y por delante de PatchTST (0.2589, 4º) y DLinear (0.2654, 5º). TimeXer domina los cuatro horizontes individuales sin excepción, pero RITMO-M se mantiene a un 7.0 % del ganador en avg MSE y a un 3.7 % en avg MAE (0.2804 vs 0.2705). Es el *dataset* en el que RITMO-M obtiene la mejor posición relativa del Plan B.

<span id="_Toc229829402" class="anchor"></span>Tabla 30. Plan B — Electricity: MSE por horizonte y MSE/MAE promedio (seed única = 2021, `--features M`). Por columna, negrita marca el mejor valor y cursiva el segundo mejor.

| Modelo                           | pl=96 MSE  | pl=192 MSE | pl=336 MSE | pl=720 MSE | Avg MSE    | Avg MAE    |
|----------------------------------|------------|------------|------------|------------|------------|------------|
| **TimeXer**                      | **0.1402** | **0.1574** | **0.1762** | **0.2108** | **0.1711** | **0.2697** |
| TimeMixer                        | *0.1562*   | *0.1698*   | *0.1866*   | *0.2282*   | *0.1852*   | *0.2746*   |
| PatchTST                         | 0.1801     | 0.1874     | 0.2042     | 0.2455     | 0.2043     | 0.2942     |
| RITMO-M (hmm_soft_residual, K=4) | 0.1863     | 0.1890     | 0.2045     | 0.2460     | 0.2065     | 0.2868     |
| DLinear                          | 0.2104     | 0.2102     | 0.2231     | 0.2578     | 0.2254     | 0.3188     |

Fuente: Elaboración propia.

**En Electricity, RITMO-M ocupa la cuarta posición** en avg MSE (0.2065), por detrás de TimeXer (0.1711, 1º), TimeMixer (0.1852, 2º) y PatchTST (0.2043, 3º), y por delante de DLinear (0.2254, 5º). TimeXer domina los cuatro horizontes individuales con margen amplio (gap del 16 % al 33 % por horizonte sobre RITMO-M). El gap absoluto frente a TimeXer (+20.6 % en avg MSE) es el mayor de los cuatro *datasets* del Plan B. Aun así, RITMO-M supera a DLinear en un 8.4 %.

La lectura conjunta de las Tablas 27-30 revela tres hechos. Primero, **TimeXer gana avg MSE en los cuatro *datasets***. Segundo, **RITMO-M nunca es 1º en avg MSE pero supera a DLinear en los cuatro *datasets*** (−2.1 % ETTh1, −29.7 % ETTh2, −3.0 % Weather, −8.4 % Electricity). Tercero, **RITMO-M gana a los cuatro *baselines* en pl = 720 sobre ETTh2**. La Ilustración 18 visualiza el contraste de MSE por horizonte y la Ilustración 19 sintetiza el ranking.

<span id="_Ref228557412" class="anchor"></span>Ilustración 18. Plan B — MSE por horizonte de predicción $\mathbf{O}\  \in \ \text{\{}\mathbf{96},\ \mathbf{192},\ \mathbf{336},\ \mathbf{720}\text{\}}\ $para los cuatro datasets del Grupo 1 y los cinco modelos comparados (RITMO-M + 4 baselines SOTA), seed única (2021) (320 ejecuciones controladas).

<img src="./media/image20.png" style="width:6.29861in;height:4.48542in" />Fuente: Elaboración propia.

<span id="_Ref228557421" class="anchor"></span>Ilustración 19. Plan B — Ranking heatmap de los cinco modelos por avg MSE en cada dataset del Grupo 1 (1 = mejor, 5 = peor; seed única = 2021).

<img src="./media/image21.png" style="width:6.29861in;height:3.28056in" />Fuente: Elaboración propia.

Para cerrar el capítulo se contrasta el comportamiento de RITMO al pasar del Plan A (--features S, 3 *seeds*) al Plan B (--features M, seed única). La Tabla 31 reporta la configuración óptima, el avg MSE/MAE y la posición relativa por *dataset*, junto con la advertencia metodológica de que las magnitudes absolutas no son directamente comparables —Plan A evalúa solo la columna OT, Plan B evalúa los C canales del *dataset*—; sí lo son la variante seleccionada, el K-óptimo y la posición relativa de RITMO frente a sus respectivos *peers*.

<span id="_Ref228557436" class="anchor"></span>Tabla 31. Comparativa de RITMO entre Plan A (`--features S`, 3 seeds) y Plan B (`--features M`, seed única 2021): configuración óptima, avg MSE/MAE y posición relativa por dataset.

| Dataset     | Plan A — config         | Plan A Avg MSE | Plan A Avg MAE | Plan A rank | Plan B — config                 | Plan B Avg MSE | Plan B Avg MAE | Plan B rank |
|-------------|-------------------------|----------------|----------------|-------------|---------------------------------|----------------|----------------|-------------|
| ETTh1       | soft, K=8 (3 seeds)     | 0.0824         | 0.2242         | 5/6         | hmm_soft, K=5 (1 seed)          | 0.4506         | 0.4439         | 4/5         |
| ETTh2       | soft, K=9 (3 seeds)     | 0.2015         | 0.3548         | 1/6         | hmm_soft, K=3 (1 seed)          | 0.3967         | 0.4141         | 4/5         |
| Weather     | soft_res, K=4 (3 seeds) | 0.0016         | 0.0299         | 1/6         | hmm_soft_residual, K=5 (1 seed) | 0.2574         | 0.2804         | 3/5         |
| Electricity | soft_res, K=3 (3 seeds) | 0.3691         | 0.4361         | 3/6         | hmm_soft_residual, K=4 (1 seed) | 0.2065         | 0.2868         | 4/5         |

Fuente: Elaboración propia.

La Tabla 31 contrasta tres dimensiones del comportamiento de RITMO al pasar de S a M. La **variante seleccionada coincide en los cuatro *datasets*** (hmm_soft en ETT, hmm_soft_residual en Weather y Electricity). El **K-óptimo, en cambio, difiere en los cuatro *datasets*** (8→5 ETTh1, 9→3 ETTh2, 4→5 Weather, 3→4 Electricity) sin patrón monotónico; las Tablas 36-39 del Anexo C documentan el barrido completo. La **posición relativa de RITMO se degrada al confrontar SOTA reales**: 1º/6 → 4º/5 en ETTh2, 1º/6 → 3º/5 en Weather, 5º/6 → 4º/5 en ETTh1 y 3º/6 → 4º/5 en Electricity. La **victoria condicional en pl = 720 ETTh2 sí se replica frente a SOTA reales** (Tabla 28).

# DISCUSIÓN DE RESULTADOS

Este capítulo interpreta los resultados experimentales del Capítulo 6 en relación con la pregunta de investigación del TFG. La discusión se estructura en tres ejes: *qué demuestran* los resultados respecto a la hipótesis original, *qué no demuestran* y por tanto requiere investigación adicional, y *dónde funciona* y *dónde no funciona* la tokenización probabilística RITMO frente a las cinco técnicas determinísticas del Plan A y los cuatro *baselines* del estado del arte del Plan B.

## Qué demuestran los resultados

Los resultados del Plan A y del Plan B demuestran cuatro hallazgos principales que responden parcialmente a la pregunta de investigación.

**Los regímenes latentes de un HMM constituyen una tokenización competitiva para *forecasting* a largo plazo bajo condiciones específicas.** RITMO obtiene el primer puesto en avg MSE en dos de los cuatro *datasets* del Grupo 1 —Weather y ETTh2—, igualando o superando a las cinco técnicas determinísticas en condiciones experimentales idénticas, y queda dentro de una banda inferior al 7 % sobre el mejor *baseline* en los dos *datasets* restantes (cifras en Tablas 20-23). El resultado refuta la hipótesis nula de que un *tokenizer* probabilístico no puede competir con las representaciones determinísticas, pero acota la afirmación: la competitividad es **condicional** y depende del *dataset*. Las dos condiciones empíricas que la determinan se desarrollan en la sección 7.3.

**El comportamiento horizonte-dependiente es no-monótono y específico de cada *dataset*.** En Weather la ventaja de RITMO se concentra en horizontes cortos; en ETTh2 sucede lo contrario, con un cambio escalonado en pl = 720 que se sostiene también frente a *baselines* SOTA reales en el Plan B; en ETTh1 la posición relativa de RITMO se degrada al alargar el horizonte; y en Electricity empeora también en pl = 720. Los dos mecanismos —dominancia temprana por persistencia regimentada (Weather) y *step-change* en horizontes largos por tolerancia al *shift* (ETTh2)— son cualitativamente distintos y se discuten en la sección 7.3.

**La estructura temporal del HMM aporta valor más allá de la cuantización.** La diferencia sistemática entre RITMO y SAX-inspired —ambas técnicas con vocabulario discreto— confirma que la ventaja del HMM no proviene únicamente de mapear valores continuos a estados discretos, sino de la matriz de transición A que condiciona cada *token* en función del estado anterior. Las métricas intrínsecas de persistencia y entropía de bigramas (sección 5.6) cuantifican este efecto, y el test pareado de Wilcoxon (Tabla 24) lo confirma con significancia formal en Weather (RITMO frente a SAX-inspired) y en Electricity (RITMO frente a LLMTime-inspired).

**Frente a *baselines* SOTA reales, RITMO-M mantiene relevancia bajo la condición 2.** En modo --features M con seed única (Plan B), RITMO-M nunca alcanza la primera posición en avg MSE —TimeXer domina los cuatro *datasets*— pero (i) supera a DLinear en los cuatro *datasets* del Grupo 1 (Tablas 27-30), y (ii) **gana a los cuatro *baselines* en pl = 720 sobre ETTh2** (Tabla 28). Esta segunda victoria valida frente a SOTA reales el mecanismo de tolerancia al *shift* identificado en el Plan A, que hasta ese punto solo se había evidenciado contra *proxies* controlados.

## Qué no demuestran los resultados

Los resultados también marcan los límites de lo que los Planes A y B pueden afirmar.

**Plan B en seed única: las cifras no admiten test estadístico formal.** Por restricción de presupuesto GPU las 320 ejecuciones del Plan B se ejecutan con seed única (2021), de modo que las comparaciones entre RITMO-M y los cuatro *baselines* SOTA son **descriptivas** —rankings, márgenes relativos, conteo de victorias por horizonte— y no inferenciales, a diferencia del Plan A que sí admite Wilcoxon Bonferroni × 5 sobre 12 pares por *dataset* (Tabla 24). En particular, la robustez de la victoria condicional de RITMO-M en pl = 720 ETTh2 frente a múltiples *seeds* queda como pregunta abierta y se identifica como línea futura.

**Las cinco técnicas comparadas son *proxies* controlados, no implementaciones íntegras.** Las implementaciones de SAX, LLMTime, PatchTST, Autoformer y MOMENT incluidas en tecnicas/ reproducen el mecanismo de tokenización descrito por sus autores originales pero sin incorporar todos los elementos arquitectónicos auxiliares —ni el pre-entrenamiento masivo en el caso de los *foundation models*—. La comparación es válida para el objetivo del Plan A —aislar el efecto de la tokenización con un *backbone* común—, pero un revisor crítico podría señalar que no se compara contra MOMENT, LLMTime o Autoformer en sentido estricto. Esta limitación afecta únicamente al Plan A; los cuatro *baselines* del Plan B (DLinear, PatchTST, TimeMixer, TimeXer) sí son arquitecturas íntegras ejecutadas con sus hiperparámetros publicados, lo que cubre el complemento metodológico de la comparación.

**La transferencia *cross-domain* del Grupo 2 produce un resultado mixto, no una pérdida sistemática.** El Plan A evalúa la transferencia del *tokenizer* HMM congelado desde los cuatro *datasets* del Grupo 1 hacia Traffic y Exchange, y los resultados son cualitativamente distintos en los dos *targets* (Tablas 25-26): en **Traffic** la mejor fuente HMM (Weather, K = 4 *soft residual*) queda un 22.6 % por encima de MOMENT-inspired y la transferencia falla globalmente, mientras que en **Exchange** la fuente HMM ETTh1 (K = 8 *soft*) **gana avg MSE y avg MAE** sobre las cinco técnicas determinísticas reentrenadas. Esta asimetría sugiere que la pregunta correcta no es si los regímenes transfieren bien o mal, sino bajo qué condiciones específicas (sección 7.3): caracterizar con precisión qué propiedades estadísticas del *target* —periodicidad dominante, magnitud del *distribution shift*, estacionariedad— predicen el valor que cada fuente HMM puede aportar excede el alcance de este trabajo y se identifica como línea futura.

## Dónde funciona y dónde no funciona RITMO

La interpretabilidad es la propiedad diferencial de RITMO frente a todas las demás técnicas. Mientras que *patching* genera vectores de P valores crudos, los *foundation models* producen representaciones latentes de alta dimensionalidad y la descomposición separa en componentes sin significado estadístico individual, los *embeddings* de RITMO admiten inspección directa: $\mu_{k}$ indica el nivel del régimen, σₖ cuantifica su volatilidad, $A\lbrack k,\ k\rbrack\ $determina su persistencia media y $A\lbrack k,\ :\rbrack\ $codifica sus dinámicas de transición. La Ilustración 17 sobre Weather con K = 4 *soft residual* demuestra esta propiedad de forma operativa: cada *callout* asocia el estado a su centro, volatilidad, frecuencia estacionaria y auto-transición.

La diversidad de los seis *datasets* permite responder con datos cuantitativos a la pregunta más relevante del estudio: *¿en qué tipo de series y en qué horizontes la tokenización probabilística aporta ventaja sobre las técnicas determinísticas?* La hipótesis inicial del Capítulo 1 puede ahora refinarse en una **regla empírica de dos condiciones**: la tokenización HMM resulta competitiva cuando se cumple **al menos una** de las siguientes propiedades sobre el *dataset* objetivo.

**Condición 1 — Regímenes alineados (*aligned regimes*).** El *dataset* presenta regímenes con persistencia significativamente superior a la del *baseline* aleatorio (razón HMM/SAX-inspired ≥ 4×) **y** alineados con la escala predictiva del horizonte objetivo (sin dinámica multi-escala dominante). **Weather la satisface plenamente**: la persistencia HMM/SAX es 4.42×, los *runs* abarcan ~18 *timesteps* (sección 5.6), los regímenes son climáticos estacionales y los cuatro estados con K = 4 *soft residual* alinean centro y volatilidad con la estructura de la señal predictiva. RITMO gana avg MSE y avg MAE en los cuatro horizontes y supera a SAX-inspired con significancia formal (p = 0.005, Tabla 24). En el Plan B Weather sigue siendo el *dataset* en que RITMO-M obtiene la posición relativa más favorable (3º/5 frente a 5 modelos, avg MSE 0.2574 con gap del 7.0 % sobre TimeXer; Tabla 29), confirmando que la condición sobrevive parcialmente al cambio de protocolo S→M aunque la primera posición se ceda a arquitecturas SOTA optimizadas. **Electricity satisface la persistencia (4.01×) pero falla la alineación**: la dinámica predictiva es multi-escala (ciclo diurno + ciclo semanal) y una colección finita de regímenes definidos sobre la marginal de niveles no puede representar dos periodicidades superpuestas; aumentar K solo refina la partición de niveles, no separa frecuencias. RITMO queda 3º, por detrás de Autoformer-inspired (que atenúa la alta frecuencia y modela *trend* y *seasonal* en paralelo) y de SAX-inspired.

**Condición 2 — Tolerancia al *distribution shift*.** El *dataset* presenta *distribution shift* sustancial entre las particiones de entrenamiento y *test*, frente al cual un encoder paramétrico con localización por estado (μₖ, σₖ) y *posterior* *soft* γₜ(k) resulta más estable que los *breakpoints* fijos de SAX-inspired o el *patching* en escala absoluta. **ETTh2 satisface esta condición**: el *distribution shift* en la media alcanza el 46.4 % (Capítulo 4), el vocabulario K = 9 absorbe la heterogeneidad y la victoria en avg MSE se concentra en pl = 720 con un std *cross-seed* bajo (0.0066) frente al de PatchTST-inspired (0.0240). El mecanismo no es la persistencia (la razón HMM/SAX-inspired es 1.02×, prácticamente unidad) sino la invariancia de la *posterior* *soft* a desplazamientos uniformes RevIN-residuales: cada régimen carga su propio centro y volatilidad, y un cambio de nivel global se reasigna sin alterar la estructura de transiciones. Esta condición se valida frente a SOTA reales en el Plan B (Tabla 28): RITMO-M gana a los cuatro *baselines* en pl = 720 ETTh2 con MSE 0.4380, mientras que TimeXer —que domina los tres horizontes anteriores— cae al cuarto puesto en horizonte largo. El mismo mecanismo de invariancia *RevIN*-residual aporta una ventaja que las arquitecturas SOTA optimizadas no capturan, lo que extiende el alcance de la condición 2 más allá de los *proxies* controlados del Plan A.

**Fuera de ambas condiciones**, los tokenizadores deterministas mantienen la ventaja. **ETTh1** es el caso paradigmático: persistencia moderada (1.52×, sección 5.6), sin *distribution shift* dominante, dinámica predictiva en componente de baja frecuencia suave, y RITMO queda 5º (significativamente peor que PatchTST-inspired y MOMENT-inspired, p = 0.005, r = −0.97 en ambos casos según la Tabla 24). El *patching* y la descomposición *trend*–*seasonal* representan mejor la suavidad de tendencia que la partición discreta de niveles del HMM. El Plan B replica el patrón (Tabla 27): RITMO-M ocupa la 4ª posición de 5 con avg MSE 0.4506, pero las cinco columnas se concentran en una franja del 3.6 % de amplitud (0.4441-0.4603) — pérdida moderada en términos absolutos porque ningún modelo destaca claramente sobre este *dataset*.

**La regla de dos condiciones se extiende al escenario *cross-domain*** y proporciona la prueba más fuerte de su generalidad. En **Traffic** ningún *source* HMM cubre la condición de regímenes alineados —los regímenes de tráfico (*rush-hour*, fuera de pico, fin de semana) son cualitativamente distintos de los regímenes térmicos, climáticos o eléctricos del Grupo 1— y la transferencia falla globalmente: la mejor fuente HMM (Weather) queda un 22.6 % por encima de MOMENT-inspired (Tabla 25). En **Exchange**, en cambio, la condición de tolerancia al *shift* sí se cumple: la serie financiera presenta *shift* sustancial sin periodicidad clara, y la fuente HMM ETTh1 (K = 8 *soft*) actúa como encoder genérico tolerante al *shift*, ganando avg MSE (0.4624) y avg MAE (0.4708) sobre las cinco técnicas determinísticas reentrenadas (Tabla 26). Es notable que **la fuente que mejor satisface la condición *aligned regimes* *in-domain*** (Weather) **no sea la que mejor transfiere a Exchange** (donde gana ETTh1): regímenes alineados y tolerancia al *shift* son propiedades distintas, y la fuente debe seleccionarse por la condición que satisface el *target*, no por similitud de dominio.

**Como corolario, las dos condiciones predicen la elección de variante HMM.** La condición 1 (regímenes alineados) se beneficia de **soft residual**: el residual intra-régimen $r_{t} = \frac{x_{t} - \mu_{\text{soft},t}}{\sigma_{\text{soft},t}}$ aprovecha la persistencia para retener la variación continua dentro del régimen, como sucede en Weather y Electricity. La condición 2 (tolerancia al *shift*) se beneficia de **soft** sin residual: la información se concentra en las *posteriors* $\gamma_{t}(k)$ y un vocabulario amplio absorbe la heterogeneidad distributiva, como sucede en ETTh1 y ETTh2. Esta doble especialización emerge en el Plan A (Tabla 5) y se replica en el Plan B (Tabla 6) tras el cambio al modo *channel-independent*, sugiriendo que es propiedad del paradigma de tokenización y no del régimen univariado.

Esta caracterización refina la hipótesis inicial del Capítulo 1: RITMO no es ni universalmente competitivo ni universalmente inferior; **es competitivo frente a *proxies* controlados (Plan A) bajo dos condiciones empíricamente identificables —regímenes alineados o tolerancia al *shift*— y mantiene relevancia frente a *baselines* SOTA reales (Plan B) bajo la condición 2**, dominado por las alternativas determinísticas o por SOTA optimizadas en el resto. Hacer operativa esta regla constituye, junto con la interpretabilidad de los *embeddings* estructurados, la aportación principal del trabajo.

## Implicaciones para el diseño de tokenizadores probabilísticos

Cinco lecciones transversales emergen del análisis:

**1. No existe una técnica de tokenización dominante en todos los escenarios.** Este resultado es coherente con la *Accuracy Law* de Y. Wang et al. (2025): en *benchmarks* saturados, el rendimiento depende más del ajuste entre la representación y la estructura del *dataset* que de la capacidad bruta del modelo. La elección de tokenizador debería ser una decisión informada por la naturaleza del problema, no un *default* universal.

**2. La interpretabilidad no es incompatible con la competitividad.** La transparencia de los *embeddings* HMM (sección 7.3) no se obtiene a costa del rendimiento cuando el *dataset* satisface al menos una de las dos condiciones empíricas. En el Plan B esta interpretabilidad se mantiene tras el cambio al modo *channel-independent* aunque el K-óptimo difiera por *dataset* (Tabla 6), confirmando que la legibilidad de los *embeddings* es invariante al modo de *features*.

**3. El *trade-off* entre compresión y rendimiento es explícito y configurable.** La variante *hard* (Viterbi *argmax*, descrita en la sección 4.2.7 y excluida del Plan A por *gradient bottleneck*) produce ratios de compresión elevados mediante *run-length encoding* sobre la secuencia de estados, frente al ratio fijo P× del *patching* (con P = 16 en este trabajo). Las dos variantes evaluadas en el Plan A —*soft* y *soft residual*— mantienen granularidad 1:1 al ponderar los *embeddings* por las *posteriors* $\gamma_{t}(k)$ del *Forward-Backward*, priorizando la preservación de información sobre la compresión. La existencia de tres variantes con perfiles compresión/rendimiento distintos hace que la elección entre máxima compresión y máximo rendimiento sea explícita y ajustable según el caso de uso, una propiedad que las técnicas determinísticas comparadas no ofrecen de forma análoga.

**4. La transferibilidad *cross-domain* depende de la condición empírica que el *target* satisface, no del dominio del *source*.** Ajustar K y la variante HMM por *dataset* mejora el rendimiento *in-domain*, pero la transferencia *cross-domain* del *tokenizer* HMM congelado no se degrada uniformemente: en Exchange la fuente ETTh1 (K = 8 *soft*) gana avg MSE y avg MAE sobre las cinco técnicas determinísticas reentrenadas porque el *target* satisface la condición de tolerancia al *shift*, mientras que en Traffic ningún *source* HMM cubre la condición de regímenes alineados y la transferencia falla globalmente (Tablas 25-26). La elección entre tokenizaciones probabilísticas y determinísticas, por tanto, debe informarse no solo por el dominio sino por la condición empírica que el *target* presenta.

**5. La degradación del ranking al confrontar SOTA reales es gradual, no categórica.** El Plan B muestra que RITMO-M nunca alcanza la primera posición en avg MSE pero (i) supera a DLinear en los cuatro *datasets*, (ii) mantiene la posición relativa más favorable en Weather (3º/5, gap del 7 % sobre TimeXer) y (iii) replica la victoria condicional en pl = 720 ETTh2 frente a los cuatro *baselines* SOTA. La pérdida de la primera posición global se compensa con la persistencia de la ventaja específica al horizonte largo bajo *distribution shift*; la elección entre RITMO-M y arquitecturas SOTA debería informarse por el horizonte objetivo y por la presencia de *shift*, no solo por la magnitud absoluta del MSE promedio.

# CONCLUSIONES

Este capítulo cierra el trabajo sintetizando los hallazgos principales en relación con la pregunta de investigación, reconociendo de forma explícita las limitaciones del estudio que conviene mantener en el horizonte del lector y proponiendo las líneas de extensión natural que abren los resultados obtenidos.

## Conclusiones generales

El presente Trabajo Fin de Grado planteaba una pregunta concreta: *¿pueden los estados ocultos de un Hidden Markov Model actuar como tokenizador competitivo para la predicción de series temporales a largo plazo, y aportar ventajas frente a las técnicas determinísticas que dominan el estado del arte?* La evidencia experimental recogida en los Capítulos 6 y 7 permite responder de manera matizada pero positiva a esta pregunta.

La tokenización mediante HMM no es ni una alternativa marginal ni una solución universal: es competitiva bajo dos condiciones empíricamente identificables —regímenes alineados con la escala predictiva o tolerancia al *distribution shift*— y dominada por las alternativas determinísticas o por arquitecturas SOTA optimizadas fuera de ellas. En el Plan A, RITMO obtiene el primer puesto en avg MSE en Weather y ETTh2 frente a las cinco técnicas determinísticas comparadas y supera a SAX-inspired en tres de los cuatro *datasets* del Grupo 1, confirmando que la matriz de transición A aporta valor estructural más allá de la cuantización por bins. En el Plan B, RITMO-M no alcanza la primera posición —TimeXer domina los cuatro *datasets*— pero supera a DLinear en los cuatro y bate a los cuatro *baselines* en pl = 720 sobre ETTh2, validando frente a SOTA reales el mecanismo de tolerancia al *shift* identificado en condiciones controladas. Adicionalmente, los *embeddings* estructurados admiten inspección directa —cada componente tiene un significado estadístico concreto—, propiedad que las técnicas comparadas no ofrecen y que se obtiene sin coste predictivo cuando el *dataset* satisface al menos una de las dos condiciones.

La aportación principal del trabajo es haber realizado (i) **la primera comparación sistemática y controlada de seis paradigmas de tokenización sobre el mismo *backbone*** con protocolo multi-seed (Plan A), y (ii) **la primera validación de RITMO-M en modo *channel-independent* frente a cuatro *baselines* SOTA** en su configuración publicada (Plan B), identificando con datos cuantitativos las condiciones bajo las cuales la representación probabilística aporta ventaja —incluyendo la victoria condicional en horizonte largo sobre series con *distribution shift*—.

## Limitaciones

El alcance de las conclusiones debe enmarcarse en cinco limitaciones explícitas que delimitan lo que el trabajo puede afirmar y lo que requiere investigación adicional.

**1. *Proxies* controlados en el Plan A.** Las implementaciones de SAX, LLMTime, PatchTST, Autoformer y MOMENT reproducen el mecanismo de tokenización original pero no incorporan los elementos arquitectónicos auxiliares ni —en los *foundation models*— el pre-entrenamiento masivo; la limitación se circunscribe al Plan A, ya que los cuatro *baselines* del Plan B son arquitecturas íntegras ejecutadas con sus hiperparámetros publicados.

**2. Transferencia *cross-domain* asimétrica en el Grupo 2.** En Traffic la transferencia falla globalmente y en Exchange el HMM congelado supera a las cinco técnicas reentrenadas; caracterizar qué propiedades del *target* predicen el valor de cada fuente HMM excede el alcance del trabajo.

**3. Plan B en seed única (2021).** Las 320 ejecuciones controladas se ejecutan con seed única por restricción de presupuesto GPU, por lo que las comparaciones son descriptivas y no admiten test pareado de Wilcoxon; la robustez de la victoria condicional en pl = 720 ETTh2 frente a múltiples *seeds* queda como pregunta abierta.

**4. K seleccionado a horizonte único.** El barrido se ejecuta a $O\  = \ 96$ y la configuración (variante, K) se reutiliza en los tres horizontes restantes; el K seleccionado podría no ser globalmente óptimo en horizontes largos.

**5. Variante *hard* excluida *a priori*.** La variante Viterbi *argmax* queda fuera del Plan A por el cuello de botella de gradiente que introduce la asignación determinística (sección 4.2.7); un *head-to-head* formal frente a las variantes *soft* sigue siendo dimensión de sensibilidad no cubierta.

## Líneas futuras

Los resultados y limitaciones anteriores abren cuatro direcciones naturales de extensión cuyo alcance supera el marco temporal y computacional de un TFG, pero que trazan el horizonte natural de continuación del trabajo.

**1. K adaptativo y selección no paramétrica del número de regímenes.** Actualmente el número de estados K se selecciona por *dataset* mediante un barrido empírico (sección 5.3). Una extensión natural sería incorporar criterios automáticos de selección de modelo —*Bayesian Information Criterion*, *Akaike Information Criterion* o validación cruzada bayesiana— que estimen el K óptimo sin intervención manual. Una variante más ambiciosa sería emplear un HMM no paramétrico (Fox et al., 2011) en el que el número de estados se infiere del propio proceso de entrenamiento, eliminando por completo el barrido de hiperparámetros y permitiendo que cada *dataset* descubra su propia granularidad de regímenes.

**2. Extensiones multi-escala y duraciones explícitas (HSMM, jerárquicos).** El fallo en Electricity (sección 7.3) cuantifica un límite estructural del HMM de primer orden: una colección finita de regímenes definidos sobre la marginal de niveles no puede representar dos periodicidades superpuestas (ciclo diurno + ciclo semanal). Las dos respuestas naturales son (i) extender el tokenizador a un *Hidden Semi-Markov Model* —donde la distribución de duración es explícita y puede ser gaussiana, log-normal o no paramétrica (Dai et al., 2017)— relajando la suposición de duración geométrica que el HMM impone, y (ii) introducir una jerarquía de niveles (cadena de Markov con estados macro y micro, o un *sticky HDP-HMM* que separe componentes de baja y alta frecuencia). Ambas alternativas atacan directamente el contraejemplo de Electricity y son la línea más prometedora para extender la condición de *aligned regimes* a dinámicas multi-escala.

**3. Integración de RITMO como módulo de tokenización dentro de** *foundation models* **reales.** Los resultados muestran que la tokenización HMM supera o iguala a las versiones MOMENT-inspired y LLMTime-inspired —que reproducen únicamente el mecanismo de tokenización, no el pre-entrenamiento masivo— en condiciones controladas. Una línea de investigación con potencial alto consiste en sustituir el *tokenizer* de un *foundation model* completo —MOMENT pre-entrenado, Chronos o Timer— por un *tokenizer* HMM y evaluar si la combinación entre regímenes interpretables y pre-entrenamiento masivo aporta mejoras adicionales. Esta línea cae fuera del alcance de un TFG por su coste computacional, pero es la que mayor potencial ofrece para transferir la propuesta a escenarios industriales y permitiría evaluar si la interpretabilidad de los regímenes HMM es compatible con la capacidad de generalización aportada por el pre-entrenamiento a gran escala.

**4. Extensión del Plan B a multi-seed y a HMM con emisiones gaussianas multivariadas.** Los resultados del Plan B se obtienen con seed única (2021) por restricción de presupuesto GPU, lo que limita la potencia inferencial de la comparación frente a SOTA. Una extensión inmediata sería re-ejecutar las 320 ejecuciones controladas con las tres *seeds* {42, 2021, 7} del Plan A para añadir bandas de incertidumbre y test pareado de Wilcoxon, replicando el protocolo del Plan A pero con *baselines* SOTA reales. En paralelo, RITMO-M utiliza un HMM 1D *channel-independent* (sección 4.2.1) con parámetros $\left( \mu_{k},\sigma_{k},A,\pi \right)$ escalares compartidos entre canales; un HMM con emisiones gaussianas multivariadas $\mu_{k} \in \mathbb{R}^{\mathbb{C}}$ y matrices de covarianza $\Sigma_{k} \in \mathbb{R}^{\mathbb{C} \times \mathbb{C}}$ por estado podría capturar correlaciones entre canales que el modo *channel-independent* descarta, abriendo la posibilidad de regímenes inter-variables (p. ej. *rush-hour* en Traffic acoplado entre sensores adyacentes). Ambas extensiones cierran preguntas planteadas explícitamente en la sección 8.2 y son las más directamente actionables a partir del estado actual del código.

Estas cuatro líneas, junto con los hallazgos consolidados en la sección 8.1 y las limitaciones explícitas de la sección 8.2, definen el horizonte natural de continuación del trabajo y constituyen la base para una eventual transformación del TFG en una publicación científica más amplia.

# REFERENCIAS BIBLIOGRÁFICAS

Abdullahi, S., Usman Danyaro, K., Zakari, A., Abdul Aziz, I., Amila Wan Abdullah Zawawi, N., & Adamu, S. (2025). Time-Series Large Language Models: A Systematic Review of State-of-the-Art. *IEEE Access*, *13*, 30235-30261. https://doi.org/10.1109/ACCESS.2025.3535782

Abeywickrama, S., Eldele, E., Wu, M., Li, X., & Yuen, C. (2026). *Entropy Guided Dynamic Patch Segmentation for Time Series Transformers* (arXiv:2509.26157). arXiv. https://doi.org/10.48550/arXiv.2509.26157

Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S. S., Arango, S. P., Kapoor, S., Zschiegner, J., Maddix, D. C., Wang, H., Mahoney, M. W., Torkkola, K., Wilson, A. G., Bohlke-Schneider, M., & Wang, Y. (2024). *Chronos: Learning the Language of Time Series* (arXiv:2403.07815). arXiv. https://doi.org/10.48550/arXiv.2403.07815

Box, G. E., & Jenkins, G. M. (1976). *Time series analysis: Forecasting and control*. John Wiley & Sons.

Cao, D., Jia, F., Arik, S. O., Pfister, T., Zheng, Y., Ye, W., & Liu, Y. (2024). *TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting* (arXiv:2310.04948). arXiv. https://doi.org/10.48550/arXiv.2310.04948

Dai, H., Dai, B., Zhang, Y.-M., Li, S., & Song, L. (2017, febrero 6). *Recurrent Hidden Semi-Markov Model*. International Conference on Learning Representations. https://openreview.net/forum?id=HJGODLqgx

De Gooijer, J. G., & Hyndman, R. J. (2006). 25 years of time series forecasting. *International Journal of Forecasting*, *22*(3), 443-473. https://doi.org/10.1016/j.ijforecast.2006.01.001

Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum Likelihood from Incomplete Data Via the *EM* Algorithm. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, *39*(1), 1-22. https://doi.org/10.1111/j.2517-6161.1977.tb01600.x

Fox, E. B., Sudderth, E. B., Jordan, M. I., & Willsky, A. S. (2011). A sticky HDP-HMM with application to speaker diarization. *The Annals of Applied Statistics*, *5*(2A). https://doi.org/10.1214/10-AOAS395

Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., & Dubrawski, A. (2024). *MOMENT: A Family of Open Time-series Foundation Models* (arXiv:2402.03885). arXiv. https://doi.org/10.48550/arXiv.2402.03885

Gruver, N., Finzi, M., Qiu, S., & Wilson, A. G. (2024). *Large Language Models Are Zero-Shot Time Series Forecasters* (arXiv:2310.07820). arXiv. https://doi.org/10.48550/arXiv.2310.07820

Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle. *Econometrica*, *57*(2), 357. https://doi.org/10.2307/1912559

Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, *9*(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735

Jiang, Y., Pan, Z., Zhang, X., Garg, S., Schneider, A., Nevmyvaka, Y., & Song, D. (2024). *Empowering Time Series Analysis with Large Language Models: A Survey* (arXiv:2402.03182). arXiv. https://doi.org/10.48550/arXiv.2402.03182

Jibao, Z., Fu, Y., Chen, X., & Chen, G. (2025). *Inner-Instance Normalization for Time Series Forecasting* (arXiv:2510.08657). arXiv. https://doi.org/10.48550/arXiv.2510.08657

Jin, M., Wang, S., Ma, L., Chu, Z., Zhang, J. Y., Shi, X., Chen, P.-Y., Liang, Y., Li, Y.-F., Pan, S., & Wen, Q. (2024). *Time-LLM: Time Series Forecasting by Reprogramming Large Language Models* (arXiv:2310.01728). arXiv. https://doi.org/10.48550/arXiv.2310.01728

Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J. (2021, octubre 6). *Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift*. International Conference on Learning Representations. https://openreview.net/forum?id=cGDAkQo1C0p

Lai, G., Chang, W.-C., Yang, Y., & Liu, H. (2018). Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks. *The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval*, 95-104. https://doi.org/10.1145/3209978.3210006

Liang, Y., Wen, H., Nie, Y., Jiang, Y., Jin, M., Song, D., Pan, S., & Wen, Q. (2024). Foundation Models for Time Series Analysis: A Tutorial and Survey. *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, 6555-6565. https://doi.org/10.1145/3637528.3671451

Lin, J., Keogh, E., Wei, L., & Lonardi, S. (2007). Experiencing SAX: A novel symbolic representation of time series. *Data Mining and Knowledge Discovery*, *15*(2), 107-144. https://doi.org/10.1007/s10618-007-0064-z

Liu, Y., Wu, H., Wang, J., & Long, M. (2023). *Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting* (arXiv:2205.14415). arXiv. https://doi.org/10.48550/arXiv.2205.14415

Liu, Y., Zhang, H., Li, C., Huang, X., Wang, J., & Long, M. (2024). *Timer: Generative Pre-trained Transformers Are Large Time Series Models* (arXiv:2402.02368). arXiv. https://doi.org/10.48550/arXiv.2402.02368

Mensch, A., & Blondel, M. (2018). *Differentiable Dynamic Programming for Structured Prediction and Attention* (arXiv:1802.03676). arXiv. https://doi.org/10.48550/arXiv.1802.03676

Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers* (arXiv:2211.14730). arXiv. https://doi.org/10.48550/arXiv.2211.14730

Oord, A. van den, Vinyals, O., & Kavukcuoglu, K. (2018). *Neural Discrete Representation Learning* (arXiv:1711.00937). arXiv. https://doi.org/10.48550/arXiv.1711.00937

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, *12*(85), 2825-2830.

Peršak, E., Anjos, M. F., Lautz, S., & Kolev, A. (2025). *Multiple-Resolution Tokenization for Time Series Forecasting with an Application to Pricing* (arXiv:2407.03185). arXiv. https://doi.org/10.48550/arXiv.2407.03185

Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, *77*(2), 257-286. https://doi.org/10.1109/5.18626

Talukder, S., Yue, Y., & Gkioxari, G. (2025). *TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis* (arXiv:2402.16412). arXiv. https://doi.org/10.48550/arXiv.2402.16412

Tang, B., & Matteson, D. S. (2021). Probabilistic Transformer For Time Series Analysis. *Advances in Neural Information Processing Systems*, *34*, 23592-23608. https://proceedings.neurips.cc/paper/2021/hash/c68bd9055776bf38d8fc43c0ed283678-Abstract.html

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). *Attention Is All You Need* (arXiv:1706.03762). arXiv. https://doi.org/10.48550/arXiv.1706.03762

Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Ma, L., Zhang, J. Y., & Zhou, J. (2024). *TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting* (arXiv:2405.14616). arXiv. https://doi.org/10.48550/arXiv.2405.14616

Wang, Y., Wu, H., Dong, J., Qin, G., Zhang, H., Liu, Y., Qiu, Y., Wang, J., & Long, M. (2024). *TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables* (arXiv:2402.19072). arXiv. https://doi.org/10.48550/arXiv.2402.19072

Wang, Y., Wu, H., Ma, Y., Fang, Y., Zhang, Z., Liu, Y., Wang, S., Ye, Z., Xiang, Y., Wang, J., & Long, M. (2025). *Accuracy Law for the Future of Deep Time Series Forecasting* (arXiv:2510.02729). arXiv. https://doi.org/10.48550/arXiv.2510.02729

Wen, Q., Zhou, T., Zhang, C., Chen, W., Ma, Z., Yan, J., & Sun, L. (2023). *Transformers in Time Series: A Survey* (arXiv:2202.07125). arXiv. https://doi.org/10.48550/arXiv.2202.07125

Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S., & Sahoo, D. (2024). *Unified Training of Universal Time Series Forecasting Transformers* (arXiv:2402.02592). arXiv. https://doi.org/10.48550/arXiv.2402.02592

Woo, G., Liu, C., Sahoo, D., Kumar, A., & Hoi, S. (2022). *CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting* (arXiv:2202.01575). arXiv. https://doi.org/10.48550/arXiv.2202.01575

Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., & Long, M. (2023). *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis* (arXiv:2210.02186). arXiv. https://doi.org/10.48550/arXiv.2210.02186

Wu, H., Xu, J., Wang, J., & Long, M. (2022). *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting* (arXiv:2106.13008). arXiv. https://doi.org/10.48550/arXiv.2106.13008

Yeh, S.-L., & Tang, H. (2022). *Learning Dependencies of Discrete Speech Representations with Neural Hidden Markov Models* (arXiv:2210.16659). arXiv. https://doi.org/10.48550/arXiv.2210.16659

Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? *Proceedings of the AAAI Conference on Artificial Intelligence*, *37*(9), 11121-11128. https://doi.org/10.1609/aaai.v37i9.26317

Zhang, X., Chowdhury, R. R., Gupta, R. K., & Shang, J. (2024). *Large Language Models for Time Series: A Survey* (arXiv:2402.01801). arXiv. https://doi.org/10.48550/arXiv.2402.01801

Zhang, Y., & Yan, J. (2022, septiembre 29). *Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting*. The Eleventh International Conference on Learning Representations. https://openreview.net/forum?id=vSVLM2j9eie

Zhao, Y., Zhou, T., Chen, C., Sun, L., Qian, Y., & Jin, R. (2024). *Sparse-VQ Transformer: An FFN-Free Framework with Vector Quantization for Enhanced Time Series Forecasting* (arXiv:2402.05830). arXiv. https://doi.org/10.48550/arXiv.2402.05830

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting* (arXiv:2012.07436). arXiv. https://doi.org/10.48550/arXiv.2012.07436

Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., & Jin, R. (2022). *FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting* (arXiv:2201.12740). arXiv. https://doi.org/10.48550/arXiv.2201.12740

Zhou, T., Niu, P., Wang, X., Sun, L., & Jin, R. (2023). *One Fits All:Power General Time Series Analysis by Pretrained LM* (arXiv:2302.11939). arXiv. https://doi.org/10.48550/arXiv.2302.11939


# Anexo A: Análisis exploratorio de los seis *datasets*

Las cuatro ilustraciones siguientes complementan las Tablas 3-4 (sección 4.4) con la visualización gráfica del análisis exploratorio reproducible (notebooks/eda_datasets.py). Cada figura yuxtapone los seis *datasets* —ETTh1, ETTh2, Weather, Electricity, Traffic y Exchange— para facilitar la comparación cualitativa de regímenes, distribuciones, periodicidades y *distribution shift*.

<span id="_Toc229829361" class="anchor"></span>Ilustración 20. Series temporales completas de los seis datasets, con los cortes verticales que marcan las particiones train / val / test.

<img src="./media/image22.png" style="width:6.29861in;height:5.03681in" />Fuente: Elaboración propia

<span id="_Toc229829362" class="anchor"></span>Ilustración 21. Histogramas de distribución del target en cada dataset; permiten contrastar visualmente la asimetría, la dispersión y la presencia de colas.

<img src="./media/image23.png" style="width:6.29861in;height:3.68819in" />Fuente: Elaboración propia.

<span id="_Toc229829363" class="anchor"></span>Ilustración 22. Funciones de autocorrelación (ACF) hasta lag 400 para los seis datasets; revelan los picos diarios y semanales reportados en la Tabla 4.

<img src="./media/image24.png" style="width:6.29861in;height:3.43819in" />Fuente: Elaboración propia.

<span id="_Toc229829364" class="anchor"></span>Ilustración 23. Comparación visual del distribution shift entre train y test; ilustra el desplazamiento de la media y de la dispersión que motiva la doble normalización descrita en la sección 4.4.4.

<img src="./media/image25.png" style="width:6.29861in;height:2.83889in" />Fuente: Elaboración propia.

# Anexo B: Barrido de K del Plan A — MSE de validación @ $\mathbf{O\  = \ 96}$ sobre 3 *seeds*

Las cuatro tablas siguientes recogen los valores completos del barrido de K @ $O\  = \ 96$ (Plan A, modo --features S) introducido en la sección 4.7. Cada celda reporta mean ± std del MSE de validación sobre las 3 *seeds* {42, 2021, 7}. La configuración (variante, K) seleccionada por *dataset* —columna *óptimo robusto*— aparece resaltada en negrita; estas son las cuatro configuraciones HMM utilizadas en el Plan A del resto del documento (sección 4.7, Tabla 5).

<span id="_Ref228604851" class="anchor"></span>Ilustración 24. Barrido de K — Plan A: MSE de validación @ $O\  = \ 96$ frente a $K\  \in \text{\{}3,\ldots,10\text{\}}$ por dataset y variante HMM, media ± desviación sobre 3 seeds {42, 2021, 7}. Línea roja discontinua: K seleccionado (Tabla 5).

<img src="./media/image26.png" style="width:6.29861in;height:6.34653in" />Fuente: Elaboración propia.

<span id="_Toc229829404" class="anchor"></span>Tabla 32. Barrido de K — ETTh1, MSE de validación @ $O\  = \ 96$ (mean ± std sobre 3 seeds).

| K     | hmm_soft            | hmm_soft_residual |
|-------|---------------------|-------------------|
| 3     | 0.0601 ± 0.0006     | 0.0601 ± 0.0020   |
| 4     | 0.0598 ± 0.0010     | 0.0609 ± 0.0027   |
| 5     | 0.0600 ± 0.0010     | 0.0615 ± 0.0040   |
| 6     | 0.0609 ± 0.0013     | 0.0623 ± 0.0023   |
| 7     | 0.0601 ± 0.0019     | 0.0613 ± 0.0029   |
| **8** | **0.0596 ± 0.0018** | 0.0653 ± 0.0039   |
| 9     | 0.0608 ± 0.0022     | 0.0609 ± 0.0008   |
| 10    | 0.0620 ± 0.0018     | 0.0607 ± 0.0020   |

Fuente: Elaboración propia.

<span id="_Toc229829405" class="anchor"></span>Tabla 33. Barrido de K — ETTh2, MSE de validación @ $O\  = \ 96$ (mean ± std sobre 3 seeds).

| K     | hmm_soft            | hmm_soft_residual |
|-------|---------------------|-------------------|
| 3     | 0.1529 ± 0.0055     | 0.1517 ± 0.0119   |
| 4     | 0.1579 ± 0.0013     | 0.1649 ± 0.0107   |
| 5     | 0.1520 ± 0.0043     | 0.1544 ± 0.0012   |
| 6     | 0.1521 ± 0.0049     | 0.1530 ± 0.0089   |
| 7     | 0.1492 ± 0.0037     | 0.1564 ± 0.0130   |
| 8     | 0.1532 ± 0.0093     | 0.1525 ± 0.0094   |
| **9** | **0.1455 ± 0.0086** | 0.1667 ± 0.0241   |
| 10    | 0.1460 ± 0.0034     | 0.1556 ± 0.0118   |

Fuente: Elaboración propia.

<span id="_Toc229829406" class="anchor"></span>Tabla 34. Barrido de K — Weather, MSE de validación @ $O\  = \ 96$ (mean ± std sobre 3 seeds).

| K     | hmm_soft          | hmm_soft_residual     |
|-------|-------------------|-----------------------|
| 3     | 0.00130 ± 0.00004 | 0.00129 ± 0.00001     |
| **4** | 0.00127 ± 0.00002 | **0.00124 ± 0.00003** |
| 5     | 0.00144 ± 0.00010 | 0.00133 ± 0.00011     |
| 6     | 0.00136 ± 0.00008 | 0.00137 ± 0.00010     |
| 7     | 0.00133 ± 0.00005 | 0.00131 ± 0.00003     |
| 8     | 0.00130 ± 0.00009 | 0.00136 ± 0.00006     |
| 9     | 0.00139 ± 0.00012 | 0.00141 ± 0.00020     |
| 10    | 0.00137 ± 0.00013 | 0.00131 ± 0.00005     |

Fuente: Elaboración propia.

<span id="_Toc229829407" class="anchor"></span>Tabla 35. Barrido de K — Electricity, MSE de validación @ $O\  = \ 96$ (mean ± std sobre 3 seeds).

| K     | hmm_soft        | hmm_soft_residual   |
|-------|-----------------|---------------------|
| **3** | 0.3397 ± 0.0077 | **0.3100 ± 0.0062** |
| 4     | 0.3408 ± 0.0115 | 0.3134 ± 0.0250     |
| 5     | 0.3288 ± 0.0125 | 0.3118 ± 0.0181     |
| 6     | 0.3293 ± 0.0173 | 0.3248 ± 0.0163     |
| 7     | 0.3332 ± 0.0056 | 0.3228 ± 0.0134     |
| 8     | 0.3217 ± 0.0066 | 0.3410 ± 0.0181     |
| 9     | 0.3205 ± 0.0098 | 0.3268 ± 0.0170     |
| 10    | 0.3330 ± 0.0123 | 0.3314 ± 0.0165     |

Fuente: Elaboración propia.

# Anexo C: Barrido de K del Plan B — MSE de validación promediado sobre 4 horizontes (seed = 2021)

La Ilustración 25 visualiza el K-sweep Plan B completo y las cuatro tablas siguientes recogen sus valores numéricos. El sweep se ejecuta en modo --features M con seed única (2021) sobre los cuatro horizontes $O\  \in \text{\{}96,\ 192,\ 336,\ 720\text{\}}$ y se introduce en la sección 4.7. Cada celda reporta el MSE de validación promediado sobre los 4 horizontes para una combinación (variante, K). La configuración óptima por *dataset* —la usada como K-óptimo de RITMO-M en el resto del documento— aparece resaltada en negrita; estas son las cuatro configuraciones recogidas en la Tabla 6 de la sección 4.7.<span id="_Ref228606502" class="anchor"></span>

Ilustración 25. Barrido de K — Plan B (RITMO-M, `--features M`, seed = 2021): MSE de validación promediado sobre los 4 horizontes $O\  \in \text{\{}96,\ 192,\ 336,\ 720\text{\}}$ frente a $K\  \in \text{\{}3,\ldots,10\text{\}}$ por dataset y variante HMM. Línea roja discontinua: K seleccionado (Tabla 6).

<img src="./media/image27.png" style="width:6.29861in;height:6.31667in" />Fuente: Elaboración propia.

<span id="_Ref228609634" class="anchor"></span>Tabla 36. Barrido de K — ETTh1, Plan B: MSE de validación promedio sobre 4 horizontes (seed = 2021).

| K     | hmm_soft   | hmm_soft_residual |
|-------|------------|-------------------|
| 3     | 0.4800     | 0.5436            |
| 4     | 0.4707     | 0.4745            |
| **5** | **0.4506** | 0.4767            |
| 6     | 0.4729     | 0.4828            |
| 7     | 0.4621     | 0.4856            |
| 8     | 0.4575     | 0.5166            |
| 9     | 0.4699     | 0.5026            |
| 10    | 0.4605     | 0.5026            |

Fuente: Elaboración propia.

<span id="_Toc229829409" class="anchor"></span>Tabla 37. Barrido de K — ETTh2, Plan B: MSE de validación promedio sobre 4 horizontes (seed = 2021).

| K     | hmm_soft   | hmm_soft_residual |
|-------|------------|-------------------|
| **3** | **0.3967** | 0.3982            |
| 4     | 0.4111     | 0.4032            |
| 5     | 0.4089     | 0.4063            |
| 6     | 0.4026     | 0.4041            |
| 7     | 0.4206     | 0.4180            |
| 8     | 0.4131     | 0.4124            |
| 9     | 0.4158     | 0.4068            |
| 10    | 0.4133     | 0.4131            |

Fuente: Elaboración propia.

<span id="_Toc229829410" class="anchor"></span>Tabla 38. Barrido de K — Weather, Plan B: MSE de validación promedio sobre 4 horizontes (seed = 2021).

| K     | hmm_soft | hmm_soft_residual |
|-------|----------|-------------------|
| 3     | 0.2804   | 0.2584            |
| 4     | 0.2677   | 0.2608            |
| **5** | 0.2653   | **0.2574**        |
| 6     | 0.2662   | 0.2602            |
| 7     | 0.2673   | 0.2597            |
| 8     | 0.2630   | 0.2579            |
| 9     | 0.2634   | 0.2585            |
| 10    | 0.2621   | 0.2588            |

Fuente: Elaboración propia.

<span id="_Toc229829411" class="anchor"></span>Tabla 39. Barrido de K — Electricity, Plan B: MSE de validación promedio sobre 4 horizontes (seed = 2021).

| K     | hmm_soft | hmm_soft_residual |
|-------|----------|-------------------|
| 3     | 0.2211   | 0.2078            |
| **4** | 0.2168   | **0.2065**        |
| 5     | 0.2151   | 0.2073            |
| 6     | 0.2128   | 0.2067            |
| 7     | 0.2113   | 0.2071            |
| 8     | 0.2107   | 0.2069            |
| 9     | 0.2112   | 0.2078            |
| 10    | 0.2098   | 0.2073            |

Fuente: Elaboración propia.

# Anexo D: Comparativa visual de predicciones por técnica — Electricity pl = 96

Las seis ilustraciones siguientes muestran la predicción de cada una de las seis técnicas del Plan A sobre **la misma muestra del conjunto de test** (índice 2043) de Electricity con horizonte pl = 96 y seed = 2021. Todos los paneles comparten escala Y para permitir contraste visual directo. La línea continua azul representa el *ground truth*; la línea discontinua naranja, la predicción.

En esta muestra concreta —seleccionada por presentar estructura temporal pronunciada (ground truth con rango 2 933–4 777, σ ≈ 510, 19 cambios de signo en la derivada) y por **dominio de RITMO en MSE y en** $\mathbf{R}^{\mathbf{2}}$ **frente a las cinco técnicas determinísticas**— RITMO obtiene el menor MSE y el mayor $R^{2}$ con un margen sustancial. RITMO explica el 89 % de la varianza del *ground truth* ($R^{2} = 0.894$), mientras que el mejor *baseline* (Autoformer-inspired) se queda en $R^{2}\  = \ 0.600$ y los demás se sitúan entre 0.292 y 0.514. La diferencia visual entre RITMO y los *baselines* es directamente atribuible a la matriz de transición A: el HMM identifica el régimen y la posición intra-régimen, mientras que los *tokens* determinísticos no pueden seguir la magnitud del ciclo.

<span id="_Toc229829367" class="anchor"></span>Ilustración 26. RITMO (HMM, K=3 soft residual) — predicción sobre Electricity pl = 96 (sample 2043, seed = 2021); MSE de muestra = 27 481.59 ($R^{2} = 0.894$).

<img src="./media/image28.png" style="width:6.29861in;height:2.06458in" />Fuente: Elaboración propia.

<span id="_Toc229829368" class="anchor"></span>Ilustración 27. PatchTST-inspired — predicción sobre Electricity pl = 96 (sample 2043, seed = 2021); MSE de muestra = 160 779.46 ($R^{2} = 0.381$).

<img src="./media/image29.png" style="width:6.29861in;height:2.06458in" />Fuente: Elaboración propia.

<span id="_Toc229829369" class="anchor"></span>Ilustración 28. MOMENT-inspired — predicción sobre Electricity pl = 96 (sample 2043, seed = 2021); MSE de muestra = 126 217.33 ($R^{2} = 0.514$).

<img src="./media/image30.png" style="width:6.29861in;height:2.06458in" />Fuente: Elaboración propia.

<span id="_Toc229829370" class="anchor"></span>Ilustración 29. Autoformer-inspired — predicción sobre Electricity pl = 96 (sample 2043, seed = 2021); MSE de muestra = 103 995.95 ($R^{2} = 0.600$).

<img src="./media/image31.png" style="width:6.29861in;height:2.06458in" />Fuente: Elaboración propia.

<span id="_Toc229829371" class="anchor"></span>Ilustración 30. LLMTime-inspired — predicción sobre Electricity pl = 96 (sample 2043, seed = 2021); MSE de muestra = 131 375.16 ($R^{2} = 0.494$).

<img src="./media/image32.png" style="width:6.29861in;height:2.06458in" />Fuente: Elaboración propia.

<span id="_Toc229829372" class="anchor"></span>Ilustración 31. SAX-inspired — predicción sobre Electricity pl = 96 (sample 2043, seed = 2021); MSE de muestra = 184 054.47 ($R^{2} = 0.292$).

<img src="./media/image33.png" style="width:6.29861in;height:2.06458in" />Fuente: Elaboración propia.
