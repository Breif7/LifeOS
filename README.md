# Sistema Cognitivo LifeOS-ML
### Planificación por Sprints Quincenales, Control Presupuestal Automatizado y Gestión del Valor Ganado (EVM) con Machine Learning Aplicado

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/pytest-23%20passed%20(100%25)-brightgreen.svg)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/breif7/LifeOS/blob/main/LifeOS_ML_EVM_Sprint_Tracker_Colab.ipynb)

**Código del Proyecto:** `PRJ-LIFEOS-ML-2026-Q4`  
**Director del Proyecto y Autor:** Ing. Miguel Martínez Ortega (*Ingeniero en Nanotecnología · Maestría en Inteligencia Artificial, UNIR*)  
**Posicionamiento:** Portafolio Técnico Demostrativo para roles de **Machine Learning Engineer / AI Tech Lead**.

---

## 📌 1. Visión General y Gobernanza

**LifeOS-ML** es una solución de ingeniería que sustituye el seguimiento reactivo y manual de proyectos y finanzas personales por un sistema gobernado cuantitativamente bajo un enfoque metodológico híbrido:
- **Línea Base Predictiva:** Guía PMBOK® 6ª y 7ª Edición, EDT/WBS de 16 paquetes, Método del Camino Crítico (CPM/PERT) y Gestión del Valor Ganado (EVM).
- **Ejecución Ágil:** Scrum con Sprints Quincenales (15 días naturales / 10 días hábiles) sincronizados con los días de dispersión de nómina (días 1 y 16) y Tablero Kanban con límite estricto de **WIP = 3**.

---

## 📐 2. La Triple Restricción (Win Conditions)

| Dimensión | Especificación Oficial de la Línea Base |
| :--- | :--- |
| **Alcance (Scope)** | 16 paquetes de trabajo en 5 fases (EDT/WBS), motor algorítmico CPM/PERT, clasificador NLP supervisado en NumPy puro (Matriz de Covey), control EVM dual y regresión predictiva Ridge L2. |
| **Tiempo (Schedule)** | 12 Semanas (6 Sprints Quincenales · 60 días hábiles). Presupuesto hasta la Conclusión en esfuerzo: **$BAC = 240.0$ horas técnicas** (4 horas diarias de trabajo profundo). |
| **Coste (Budget)** | Ingreso quincenal: **$14,691.00 MXN**. Línea Base de Gasto Operativo (**$BAC = \$10,900.00$ MXN/quincena** · $\$65,400.00$ MXN en 6Q). Fondo de amortización y liquidez: **$3,791.00 MXN/quincena**. Tope discrecional diario: **$300.00 MXN**. |
| **Calidad (Quality)** | F1-Score macro $> 0.85$ en NLP (alcanzado: **0.955**), exactitud CV 4-Fold $> 95.0%$ (alcanzado: **96.0%**), y **0% de recargos por intereses bancarios** (ahorro garantizado con CAT 0%). |

---

## 🧠 3. Motores de Machine Learning Aplicados (Enfoque ML Engineer)

### 🔹 Motor 1: Clasificador NLP Supervisado (`NumpyTfidfSoftmaxClassifier`)
- **Implementación:** Programado desde cero en **NumPy puro** sin dependencias externas de inferencia.
- **Vectorizador:** `SublinearTfidfVectorizer` con unigramas y bigramas de palabras, tetragramas de caracteres morfológicos ($c:<pago>$), escalado sublineal $1 + \ln(tf)$ y normalización $L_2$.
- **Clasificador:** Softmax Multinomial con Regularización $L_2$ optimizado mediante **Adam**.
- **Regla Anti-Scope Creep (Tema 3.4):** Bloqueo automático para actividades catalogadas en el **Cuadrante IV de Covey (No Urgente y No Importante)** o fuera del catálogo de la EDT.
- **Rendimiento:**
  - Exactitud en Validación Cruzada Estratificada (4-Fold CV): **96.03%**
  - Macro F1-Score: **0.9548**

### 🔹 Motor 2: Predictor EVM No Lineal (`PolynomialRidgeRegressor`)
- Modela la no linealidad en productividad generada por el pico de 48 horas en el Sprint 5 (exámenes y entregas finales de maestría UNIR), superando la proyección lineal simplista del PMBOK ($EAC = BAC / CPI$).
- Vector de entrada: $x_t = [\text{Sprint}_t, PV_t, CPI_{t-1}, SPI_{t-1}]$ expandido a características polinomiales de grado 2.
- Solución exacta de Ridge: $\theta = (X^T X + \lambda I)^{-1} X^T y$.
- Proyección al corte de Quincena 3:
  - **$EAC_{\text{típico}}$ (Lineal PMBOK):** $233.68\text{ h}$ / $\$63,699.91\text{ MXN}$
  - **$EAC_{\text{ML}}$ (Ridge Polinomial):** **$235.63\text{ h}$** / **$\$64,120.00\text{ MXN}$**

---

## 📊 4. Métricas de Valor Ganado al Corte de Quincena 3 (Mitad del Proyecto)

| Indicador EVM | Dimensión Técnica (Horas) | Dimensión Financiera (MXN) | Estado / Interpretación |
| :--- | :---: | :---: | :--- |
| **BAC** (Presupuesto Conclusión) | 240.00 h | $65,400.00 MXN | Línea base congelada para 6 quincenas |
| **PV** (Valor Planificado) | 114.00 h | $32,700.00 MXN | Programado al cierre de Q3 |
| **EV** (Valor Ganado) | 114.00 h | $32,700.00 MXN | 100% de paquetes S1 a S3 concluidos |
| **AC** (Costo/Esfuerzo Real) | 111.00 h | $31,850.00 MXN | Gasto y esfuerzo real incurrido |
| **CV** (Variación de Coste) | **+3.00 h** | **+$850.00 MXN** | **Favorable:** Ahorro neto acumulado |
| **SV** (Variación Cronograma) | **0.00 h** | **$0.00 MXN** | **En plazo exacto** sobre ruta crítica |
| **CPI** (Índice Desempeño Coste) | **1.0270** | **1.0267** | **Sobresaliente (> 1.0)** |
| **SPI** (Índice Desempeño Plazo) | **1.0000** | **1.0000** | **Alineación perfecta (= 1.0)** |
| **CSI** (Índice Coste-Cronograma) | **1.0270** | **1.0267** | Alta probabilidad de conclusión exitosa |
| **EAC_ML** (Ridge Regressor) | **235.63 h** | **$64,120.00 MXN** | Ajuste realista por sobrecarga en Q5 |
| **VAC** (Variación a Conclusión) | **+6.32 h** | **+$1,700.09 MXN** | Ahorro disponible para pago acelerado de deuda |

---

## 🛠️ 5. Estructura del Proyecto

```text
LifeOS/
├── LifeOS_ML_EVM_Sprint_Tracker_Colab.ipynb  # Notebook maestro reproducible en Google Colab
├── PGP_Portafolio_LifeOS_ML_Engineer.pdf    # Documento Maestro PGP ("Biblia del Proyecto")
├── requirements.txt                         # Dependencias del proyecto
├── README.md                                # Documentación de ingeniería y arquitectura
├── build_notebook.py                        # Script generador del notebook Colab
├── lifeos/                                  # Código fuente modular
│   ├── __init__.py                          # Exportaciones del paquete
│   ├── config.py                            # Catálogo WBS (16 paquetes), 10 tarjetas y 6 riesgos
│   ├── cpm_pert.py                          # Motor PDM, Camino Crítico y Monte Carlo (10k iteraciones)
│   ├── nlp_classifier.py                    # Sublinear TF-IDF + Softmax Classifier en NumPy puro
│   ├── evm_predictor.py                     # Tracker EVM y Regresión Polinomial Ridge L2
│   ├── finance_manager.py                   # Cronograma de 10 tarjetas, nómina y tope $300/día
│   ├── forge3d.py                           # Cotizador de lotes 3D y mantenimiento Anycubic Kobra S1
│   └── api.py                               # Microservicio REST en FastAPI
└── tests/                                   # Suite de pruebas unitarias automatizadas
    ├── __init__.py
    ├── test_cpm_pert.py                     # Validación de holguras, ruta crítica y PERT
    ├── test_nlp_classifier.py                 # Validación de inferencia, bloqueo Q4 y CV > 95%
    ├── test_evm_predictor.py                  # Validación de fórmulas EVM y regresión Ridge
    ├── test_finance_manager.py              # Validación de cobertura quincenal y tope $300
    ├── test_forge3d.py                      # Validación de cotizaciones 3D y mantenimiento
    └── test_api.py                          # Validación de endpoints REST de FastAPI
```

---

## 🚀 6. Instalación y Ejecución

### 1. Clonar el repositorio y crear entorno virtual
```bash
git clone https://github.com/breif7/LifeOS.git
cd LifeOS
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Ejecutar la suite de pruebas unitarias (23 tests)
```bash
pytest tests/ -v
```
Resultado garantizado:
```text
======================= 23 passed in 17.24s =======================
```

### 3. Ejecutar el microservicio FastAPI
```bash
uvicorn lifeos.api:app --reload --port 8000
```
Visita la documentación interactiva Swagger en: `http://localhost:8000/docs`.

### 4. Ejecutar en Google Colab
Abre directamente [`LifeOS_ML_EVM_Sprint_Tracker_Colab.ipynb`](LifeOS_ML_EVM_Sprint_Tracker_Colab.ipynb) en Google Colab para ejecutar todas las celdas interactivas con gráficos de Curvas S, histogramas de Monte Carlo y clasificación NLP en tiempo real.

---

## 💼 7. Narrativa de Empleabilidad (Método STAR para Entrevistas de ML Engineer)

- **Situación:** Durante mi Maestría en Inteligencia Artificial en UNIR, requería coordinar simultáneamente el desarrollo de software y modelos de ML para mi portafolio profesional, los entregables académicos y un estricto control presupuestal quincenal sin sobrecostes ni retrasos.
- **Tarea:** Diseñar y construir un sistema de ingeniería reproducible (**LifeOS-ML**) que llevara la gestión de proyectos desde un enfoque artesanal hacia un estándar industrial (PMBOK® 7 + Scrum/Kanban + EVM), automatizando la priorización de tareas/gastos y el pronóstico de tiempos mediante Machine Learning.
- **Acción:** Estructuré un Plan de Gestión del Proyecto (PGP) con una EDT de 16 paquetes a lo largo de 6 Sprints quincenales (240 h). Programé en Python un motor de Camino Crítico (CPM) con simulación Monte Carlo (10,000 iteraciones), entrené un clasificador NLP puro en NumPy (TF-IDF sublineal + Softmax L2) para mapear registros en lenguaje natural hacia la EDT y la Matriz de Covey, e implementé una Regresión Polinomial Ridge L2 acoplada a un tablero de Valor Ganado.
- **Resultado:** El clasificador NLP superó **96.0% de exactitud en 4-Fold CV y macro F1 > 0.95**; la reasignación hacia el Cuadrante II elevó el tiempo de ingeniería profunda del **32% al 68%** semanal; y el control por Valor Ganado mantuvo el proyecto en **plazo exacto ($SPI = 1.00$) y bajo presupuesto tanto en esfuerzo ($CPI = 1.027$) como en finanzas ($CPI = 1.027$)**, ahorrando **$1,700 MXN** adicionales y garantizando **0% de intereses bancarios**.

---
*(c) 2026 Ing. Miguel Martínez Ortega · Licencia MIT.*
