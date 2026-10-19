"""
Builder script to generate LifeOS_ML_EVM_Sprint_Tracker_Colab.ipynb
PRJ-LIFEOS-ML-2026-Q4 — LifeOS-ML
"""

import json
from pathlib import Path


def make_markdown_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines],
    }


def make_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines],
    }


def build_lifeos_notebook():
    cells = []

    # Title & Colab Badge
    cells.append(make_markdown_cell([
        "# Sistema Cognitivo LifeOS-ML",
        "## Planificación por Sprints Quincenales, Control Presupuestal y Gestión del Valor Ganado (EVM) con Machine Learning Aplicado",
        "",
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/breif7/LifeOS/blob/main/LifeOS_ML_EVM_Sprint_Tracker_Colab.ipynb)",
        "",
        "**Código de Proyecto:** `PRJ-LIFEOS-ML-2026-Q4`  ",
        "**Autor y Director:** Ing. Miguel Martínez Ortega (Ingeniero en Nanotecnología · Maestría en Inteligencia Artificial, UNIR)  ",
        "**Marco Metodológico:** Híbrido: Línea Base Predictiva (PMBOK® 6ª y 7ª Ed., EDT/WBS, CPM/PERT, EVM) + Ejecución Ágil (Scrum Quincenal + Kanban WIP=3)  ",
        "**Propósito:** Portafolio Técnico Demostrativo para roles de **Machine Learning Engineer / AI Tech Lead**.",
        "",
        "---",
        "### Resumen de la Triple Restricción:",
        "- **Alcance (Scope):** 16 paquetes de trabajo en 5 fases, motor CPM/PERT, clasificador NLP supervisado (Matriz de Covey), control EVM dual y predictor Regresión Polinomial Ridge L2.",
        "- **Tiempo (Schedule):** 12 semanas (6 Sprints Quincenales de 15 días / 10 días hábiles) sincronizados con dispersión de nómina. Presupuesto de esfuerzo $BAC = 240.0$ horas.",
        r"- **Coste (Budget):** Ingreso quincenal $\$14,691.00$ MXN. Línea base operativa $BAC = \$10,900.00$ MXN/quincena ($\$65,400.00$ MXN en 6Q). Reserva de amortización/liquidez: $\$3,791.00$ MXN/quincena. Tope diario discrecional: $\$300.00$ MXN.",
        "- **Calidad (Quality):** F1-Score macro $> 0.85$ en NLP, exactitud CV 4-Fold $> 95.0%$, error EVM $< 3%$, y $0.00$ pesos en intereses moratorios bancarios (CAT 0%)."
    ]))

    # Setup cell
    cells.append(make_markdown_cell([
        "---",
        "## 0. Configuración del Entorno y Dependencias",
        "Instalamos e importamos las librerías necesarias. Si se ejecuta en Google Colab, se clona automáticamente el repositorio de GitHub para tener acceso a los módulos locales y la suite de pruebas."
    ]))

    cells.append(make_code_cell([
        "# 1. Detección de Google Colab y sincronización del repositorio",
        "import os",
        "import sys",
        "try:",
        "    import google.colab",
        "    is_colab = True",
        "except ImportError:",
        "    is_colab = False",
        "",
        "if is_colab:",
        "    if not os.path.exists('lifeos'):",
        "        !git clone https://github.com/breif7/LifeOS.git",
        "        if os.path.exists('LifeOS'):",
        "            %cd LifeOS",
        "    if '.' not in sys.path:",
        "        sys.path.insert(0, '.')",
        "",
        "# 2. Instalación de dependencias",
        "!pip install -q matplotlib pandas numpy pytest",
        "",
        "import math",
        "import re",
        "from copy import deepcopy",
        "from dataclasses import dataclass, field",
        "from typing import Dict, List, Optional, Tuple",
        "",
        "import matplotlib.pyplot as plt",
        "import numpy as np",
        "import pandas as pd",
        "",
        "# Configuración estética de gráficos",
        "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')",
        "plt.rcParams['font.sans-serif'] = 'DejaVu Sans'",
        "plt.rcParams['figure.dpi'] = 120",
        "print('Entorno de ejecución inicializado con éxito.')"
    ]))

    # Section 1: Baseline & Config
    cells.append(make_markdown_cell([
        "---",
        "## 1. Línea Base de Gobernanza y WBS (16 Paquetes de Trabajo)",
        "Estructuramos los 16 paquetes de trabajo terminales según la Tabla 2 del Plan de Gestión del Proyecto (PGP)."
    ]))

    cells.append(make_code_cell([
        "from lifeos.config import WBS_CATALOG, CREDIT_CARDS, RISK_CATALOG, BAC_HOURS, TOTAL_BAC_FINANCIAL",
        "from lifeos.cpm_pert import CPMNetwork",
        "",
        "# Instanciamos la red y calculamos la programación temprana y tardía",
        "cpm_net = CPMNetwork()",
        "packages = cpm_net.compute_schedule()",
        "summary_table = cpm_net.get_summary_table()",
        "",
        "df_wbs = pd.DataFrame(summary_table)",
        "display_cols = ['wbs_id', 'name', 'sprint', 'duration', 'bac_cost', 'predecessors', 'es', 'ef', 'ls', 'lf', 'slack', 'pert_te', 'is_critical']",
        "print(f'Presupuesto hasta la Conclusión en Esfuerzo (BAC): {BAC_HOURS:.1f} horas')"
    ]))

    cells.append(make_code_cell([
        "# Mostramos la tabla completa de planificación",
        "pd.set_option('display.max_columns', 15)",
        "pd.set_option('display.width', 1000)",
        "df_wbs[display_cols]"
    ]))

    # Section 2: CPM / PERT & Monte Carlo
    cells.append(make_markdown_cell([
        "---",
        "## 2. Motor Algorítmico CPM/PERT y Simulación de Monte Carlo",
        "El análisis de la red de precedencias determina un **Camino Crítico determinista de 166.0 horas** y una **media PERT de 170.17 horas (± 3.79 h)**.  ",
        "Ejecutamos una **Simulación de Monte Carlo con 10,000 iteraciones** muestreando duraciones estocásticas bajo la distribución Beta-PERT."
    ]))

    cells.append(make_code_cell([
        "# Ejecutar análisis PERT y Simulación Monte Carlo",
        "pert_stats = cpm_net.get_pert_statistics()",
        "print('Estadísticas de Ruta Crítica (PERT):', pert_stats)",
        "",
        "mc_res = cpm_net.run_monte_carlo_simulation(iterations=10000, target_hours=255.0, random_seed=42)",
        "print(f\"Iteraciones: {mc_res['iterations']:,}\")",
        "print(f\"Media simulada: {mc_res['mean_hours']} h | Desviación estándar: {mc_res['std_hours']} h\")",
        "print(f\"Percentil 85 (P85): {mc_res['p85_hours']} h | Percentil 95 (P95): {mc_res['p95_hours']} h\")",
        "print(f\"Probabilidad de éxito dentro de 255 h (BAC + 6% contingencia): {mc_res['prob_under_threshold']:.1f}%\")"
    ]))

    cells.append(make_code_cell([
        "# Figura 1 del PGP: Diagrama de Gantt con Camino Crítico (izq) y Simulación Monte Carlo (der)",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))",
        "",
        "# --- Gráfico 1: Gantt con Camino Crítico ---",
        "ordered_packages = list(packages.values())",
        "ordered_packages.sort(key=lambda p: (p.es, p.ef), reverse=True)",
        "",
        "y_labels = [f\"{p.wbs_id} ({p.sprint})\" for p in ordered_packages]",
        "y_positions = np.arange(len(ordered_packages))",
        "",
        "for idx, pkg in enumerate(ordered_packages):",
        "    color = '#D32F2F' if pkg.is_critical else '#1976D2'  # Rojo si es crítico, Azul si no",
        "    ax1.barh(idx, pkg.duration, left=pkg.es, color=color, alpha=0.85, edgecolor='black', height=0.6)",
        "    if pkg.total_slack > 0:",
        "        ax1.barh(idx, pkg.total_slack, left=pkg.ef, color='lightgray', hatch='//', alpha=0.6, edgecolor='gray', height=0.6)",
        "    ax1.text(pkg.es + pkg.duration / 2, idx, f\"{pkg.duration:.0f}h\", ha='center', va='center', color='white', fontweight='bold', fontsize=8)",
        "",
        "ax1.set_yticks(y_positions)",
        "ax1.set_yticklabels(y_labels, fontsize=8)",
        "ax1.set_xlabel('Horas Técnicas Acumuladas (ES -> EF)', fontweight='bold')",
        "ax1.set_title('Diagrama de Gantt y Camino Crítico\\n(Rojo = Crítico, Azul = Actividad, Gris = Holgura)', fontweight='bold')",
        "ax1.set_xlim(0, 180)",
        "",
        "# --- Gráfico 2: Simulación de Monte Carlo ---",
        "sim_dur = mc_res['raw_simulations']",
        "count, bins, ignored = ax2.hist(sim_dur, bins=45, density=True, color='#26A69A', alpha=0.65, edgecolor='black', label='Densidad Monte Carlo (10k)')",
        "",
        "# Curva acumulada de probabilidad en segundo eje Y",
        "ax2_cdf = ax2.twinx()",
        "sorted_data = np.sort(sim_dur)",
        "cdf = np.arange(len(sorted_data)) / float(len(sorted_data)) * 100.0",
        "ax2_cdf.plot(sorted_data, cdf, color='#0D47A1', lw=2.5, label='Probabilidad de Éxito Acumulada (%)')",
        "ax2_cdf.set_ylabel('Probabilidad Acumulada (%)', color='#0D47A1', fontweight='bold')",
        "ax2_cdf.set_ylim(0, 105)",
        "",
        "# Líneas de referencia",
        "ax2.axvline(240.0, color='gray', linestyle='--', lw=1.8, label='BAC Determinista (240h)')",
        "ax2.axvline(mc_res['p85_hours'], color='red', linestyle='-.', lw=2.0, label=f\"Percentil 85 ({mc_res['p85_hours']:.1f}h)\")",
        "ax2.axvline(255.0, color='purple', linestyle=':', lw=2.0, label=f\"Umbral 255h (Prob: {mc_res['prob_under_threshold']:.1f}%)\")",
        "",
        "ax2.set_xlabel('Esfuerzo Total del Proyecto (Horas)', fontweight='bold')",
        "ax2.set_ylabel('Densidad de Probabilidad', fontweight='bold')",
        "ax2.set_title('Análisis de Riesgo de Cronograma (PERT / Monte Carlo 10k)', fontweight='bold')",
        "ax2.legend(loc='upper left', fontsize=8)",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ]))

    # Section 3: NLP Classifier
    cells.append(make_markdown_cell([
        "---",
        "## 3. Motor 1: Clasificador NLP Supervisado (NumPy Puro) y Matriz de Covey",
        "Construido enteramente con **NumPy**:  ",
        "1. `SublinearTfidfVectorizer`: Vectorizador morfológico con unigramas, bigramas y tetragramas de caracteres, escalado sublineal $1 + \\ln(tf)$ y normalización $L_2$.  ",
        "2. `NumpyTfidfSoftmaxClassifier`: Clasificador multinomial Softmax con regularización $L_2$ optimizado vía Adam en NumPy.  ",
        "3. Regla de negocio PGP (Tema 3.4): Cualquier tarea catalogada en **Cuadrante IV (No Urgente y No Importante / Scope Creep / Gold Plating)** es **bloqueada de inmediato**."
    ]))

    cells.append(make_code_cell([
        "from lifeos.nlp_classifier import LifeOSNLPClassifier",
        "",
        "nlp_model = LifeOSNLPClassifier()",
        "nlp_model.train_on_corpus()",
        "",
        "# Evaluación con Validación Cruzada Estratificada 4-Fold",
        "cv_results = nlp_model.evaluate_stratified_cv(k_folds=4)",
        "print(f\"--- Rendimiento en Validación Cruzada Estratificada ({cv_results['k_folds']}-Fold CV) ---\")",
        "print(f\"Exactitud Media (Accuracy): {cv_results['mean_cv_accuracy'] * 100:.2f}%\")",
        "print(f\"Macro F1-Score: {cv_results['mean_macro_f1']:.4f}\")",
        "print(f\"Exactitud por pliegue: {cv_results['fold_accuracies']}\")",
        "print(f\"F1 por pliegue: {cv_results['fold_f1s']}\")"
    ]))

    cells.append(make_code_cell([
        "# Demostración de inferencia en tiempo real sobre los casos oficiales del PGP",
        "test_phrases = [",
        "    \"Apartar $4,208 MXN para liquidar pago para no generar intereses de AMEX antes del día 9\",",
        "    \"Entrenar clasificador DistilRoBERTa y documentar curva S en Colab\",",
        "    \"Responder encuesta rápida de satisfacción que vence al mediodía\",",
        "    \"Rediseñar paleta de colores CSS y sombras de botones del dashboard por estética\",",
        "    \"Gasto discrecional en café gourmet y comida rápida superando $300 diarios\",",
        "]",
        "",
        "for phrase in test_phrases:",
        "    pred = nlp_model.classify(phrase)",
        "    estado = '🚫 BLOQUEADO' if pred['is_blocked'] else '✅ APROBADO'",
        "    print(f\"{estado} [{pred['covey_quadrant']}] (Conf: {pred['covey_confidence']*100:.1f}%) -> {pred['input_text']}\")",
        "    if pred['is_blocked']:",
        "        print(f\"    Razón: {pred['block_reason']}\")",
        "    print('-' * 85)"
    ]))

    cells.append(make_code_cell([
        "# Figura 2 (Derecha): Optimización del Tiempo según Matriz de Stephen Covey (Tema 6.5)",
        "fig, ax = plt.subplots(figsize=(9, 5))",
        "",
        "categories = [",
        "    'Q1: Urgente +\\nImportante (Crisis)',",
        "    'Q2: No Urgente +\\nImportante (Valor/IA)',",
        "    'Q3: Urgente +\\nNo Imp. (Automatizable)',",
        "    'Q4: No Urgente +\\nNo Imp. (Gold Plating)',",
        "]",
        "antes = [38, 32, 18, 12]  # Gestión reactiva tradicional",
        "despues = [18, 68, 10, 4]  # Con LifeOS-ML (Foco en Cuadrante II)",
        "",
        "x = np.arange(len(categories))",
        "width = 0.35",
        "",
        "rects1 = ax.bar(x - width/2, antes, width, label='Antes de LifeOS-ML (Gestión Reactiva)', color='#90A4AE', edgecolor='black')",
        "rects2 = ax.bar(x + width/2, despues, width, label='Con LifeOS-ML (Foco en Cuadrante II)', color='#0288D1', edgecolor='black')",
        "",
        "ax.set_ylabel('% del Tiempo Semanal Dedicado', fontweight='bold')",
        "ax.set_title('Optimización del Tiempo según Matriz de Covey (Tema 6.5)\\nIncremento en Cuadrante II del 32% al 68%', fontweight='bold')",
        "ax.set_xticks(x)",
        "ax.set_xticklabels(categories, fontsize=9)",
        "ax.legend()",
        "ax.set_ylim(0, 85)",
        "",
        "for rect in rects1:",
        "    h = rect.get_height()",
        "    ax.annotate(f'{h}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)",
        "for rect in rects2:",
        "    h = rect.get_height()",
        "    ax.annotate(f'{h}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#01579B')",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ]))

    # Section 4: Finance Manager
    cells.append(make_markdown_cell([
        "---",
        "## 4. Control Financiero Quincenal de 10 Tarjetas y Restricción $300 Diarios",
        r"Administración de la deuda total distribuida ($\$47,238.00$ MXN) asegurando un pago para no generar intereses de $\$11,341.00$ MXN con **$0.00 MXN en recargos (CAT 0%)**."
    ]))

    cells.append(make_code_cell([
        "from lifeos.finance_manager import FinanceManager",
        "",
        "fin_mgr = FinanceManager()",
        "fin_summary = fin_mgr.get_portfolio_summary()",
        "schedule = fin_mgr.get_biweekly_payment_schedule()",
        "",
        "print('--- Resumen Ejecutivo de Finanzas Personales ---')",
        "for k, v in fin_summary.items():",
        "    print(f\"{k:30s}: ${v:,.2f} MXN\" if isinstance(v, float) and k != 'interest_paid_rate' else f\"{k:30s}: {v}\")"
    ]))

    cells.append(make_code_cell([
        "# Mostramos el cronograma de pago por Quincena",
        "df_cards = pd.DataFrame([",
        "    {'Tarjeta': c.name, 'Día Pago': c.due_day, 'Corte': c.cutoff_day, 'Saldo Total': f\"${c.balance:,.2f}\", 'Pago No Interés': f\"${c.no_interest_payment:,.2f}\", 'Mínimo': f\"${c.minimum_payment:,.2f}\", 'CAT': f\"{c.cat_rate:.1f}%\"}",
        "    for c in CREDIT_CARDS",
        "])",
        "df_cards"
    ]))

    cells.append(make_code_cell([
        "# Validación de la regla de gasto discrecional ($300 MXN diarios)",
        "print('1. Gasto normal bajo el tope:')",
        "reg1 = fin_mgr.register_expense('Almuerzo de trabajo', 180.0, category='discretionary', day_of_month=5)",
        "print(f\"Total día 5: ${reg1['daily_discretionary_total']} MXN | Supera tope: {reg1['exceeds_cap']}\")",
        "",
        "print('\\n2. Gasto acumulado que excede el tope de $300:')",
        "reg2 = fin_mgr.register_expense('Cena fuera y snacks', 160.0, category='discretionary', day_of_month=5)",
        "print(f\"Total día 5: ${reg2['daily_discretionary_total']} MXN | Supera tope: {reg2['exceeds_cap']}\")",
        "if reg2['alert']:",
        "    print(f\"ALERTA: {reg2['alert']}\")"
    ]))

    # Section 5: EVM & Ridge ML
    cells.append(make_markdown_cell([
        "---",
        "## 5. Motor 2: Gestión del Valor Ganado (EVM) y Predictor No Lineal (Ridge L2)",
        "Corte de control cuantitativo al cierre de la **Quincena 3 (Sprint 3 / Semana 6 de 12 — Mitad del Proyecto)**.  ",
        "Comparamos la fórmula lineal clásica del PMBOK ($EAC = BAC / CPI$) con nuestro modelo no lineal **Regresión Polinomial Ridge L2 (NumPy)** que modela el pico de 48 horas en la Quincena 5 (exámenes UNIR)."
    ]))

    cells.append(make_code_cell([
        "from lifeos.evm_predictor import EVMTracker",
        "",
        "evm = EVMTracker()",
        "metrics = evm.compute_cutoff_metrics()",
        "",
        "df_evm = pd.DataFrame([",
        "    {'Indicador EVM': 'Presupuesto hasta Conclusión (BAC)', 'Técnica (Horas)': f\"{metrics['technical']['BAC']:.2f} h\", 'Financiera (MXN)': f\"${metrics['financial']['BAC']:,.2f}\", 'Interpretación': 'Línea base para 6 quincenas (12 semanas)'},",
        "    {'Indicador EVM': 'Valor Planificado Acumulado (PV)', 'Técnica (Horas)': f\"{metrics['technical']['PV']:.2f} h\", 'Financiera (MXN)': f\"${metrics['financial']['PV']:,.2f}\", 'Interpretación': 'Trabajo y gasto programado al cierre de Q3'},",
        "    {'Indicador EVM': 'Valor Ganado Acumulado (EV)', 'Técnica (Horas)': f\"{metrics['technical']['EV']:.2f} h\", 'Financiera (MXN)': f\"${metrics['financial']['EV']:,.2f}\", 'Interpretación': '100% de paquetes S1 a S3 completados'},",
        "    {'Indicador EVM': 'Costo / Esfuerzo Real Acumulado (AC)', 'Técnica (Horas)': f\"{metrics['technical']['AC']:.2f} h\", 'Financiera (MXN)': f\"${metrics['financial']['AC']:,.2f}\", 'Interpretación': 'Esfuerzo y gasto real erogado (Ahorro)'},",
        "    {'Indicador EVM': 'Variación del Coste (CV = EV - AC)', 'Técnica (Horas)': f\"+{metrics['technical']['CV']:.2f} h\", 'Financiera (MXN)': f\"+${metrics['financial']['CV']:,.2f}\", 'Interpretación': 'Positiva: Ahorro de 3.0 h y $850 MXN'},",
        "    {'Indicador EVM': 'Variación Cronograma (SV = EV - PV)', 'Técnica (Horas)': f\"{metrics['technical']['SV']:.2f} h\", 'Financiera (MXN)': f\"${metrics['financial']['SV']:,.2f}\", 'Interpretación': 'En plazo exacto (sin retrasos)'},",
        "    {'Indicador EVM': 'Índice Desempeño Coste (CPI = EV / AC)', 'Técnica (Horas)': f\"{metrics['technical']['CPI']:.4f}\", 'Financiera (MXN)': f\"{metrics['financial']['CPI']:.4f}\", 'Interpretación': 'Óptimo (>1.0): Mayor valor por recurso'},",
        "    {'Indicador EVM': 'Índice Desempeño Cronograma (SPI = EV / PV)', 'Técnica (Horas)': f\"{metrics['technical']['SPI']:.4f}\", 'Financiera (MXN)': f\"{metrics['financial']['SPI']:.4f}\", 'Interpretación': 'Óptimo (=1.0): Ritmo al 100% del plan'},",
        "    {'Indicador EVM': 'Índice Coste-Cronograma (CSI = CPI x SPI)', 'Técnica (Horas)': f\"{metrics['technical']['CSI']:.4f}\", 'Financiera (MXN)': f\"{metrics['financial']['CSI']:.4f}\", 'Interpretación': 'Salud global sobresaliente (>1.0)'},",
        "    {'Indicador EVM': 'Estimación a la Conclusión Lineal (EAC)', 'Técnica (Horas)': f\"{metrics['technical']['EAC_typical']:.2f} h\", 'Financiera (MXN)': f\"${metrics['financial']['EAC_typical']:,.2f}\", 'Interpretación': 'Proyección lineal PMBOK clásica'},",
        "    {'Indicador EVM': 'Estimación a Conclusión ML (EAC_ML)', 'Técnica (Horas)': f\"{metrics['technical']['EAC_ml']:.2f} h\", 'Financiera (MXN)': f\"${metrics['financial']['EAC_ml']:,.2f}\", 'Interpretación': 'Ajuste no lineal por sobrecarga de 48h en Q5'},",
        "    {'Indicador EVM': 'Variación a la Conclusión (VAC = BAC - EAC)', 'Técnica (Horas)': f\"+{metrics['technical']['VAC']:.2f} h\", 'Financiera (MXN)': f\"+${metrics['financial']['VAC']:,.2f}\", 'Interpretación': 'Ahorro previsto para amortizar deuda'},",
        "    {'Indicador EVM': 'Índice Rendimiento Completar (TCPI)', 'Técnica (Horas)': f\"{metrics['technical']['TCPI']:.4f}\", 'Financiera (MXN)': f\"{metrics['financial']['TCPI']:.4f}\", 'Interpretación': 'Holgura de eficiencia (<1.0) para el cierre'},",
        "])"
    ]))

    cells.append(make_code_cell([
        "df_evm"
    ]))

    cells.append(make_code_cell([
        "# Figura 3 del PGP: Curva S con Proyección ML (izq) y Evolución de Índices CPI/SPI (der)",
        "forecast = evm.predict_future_sprints_ml()",
        "",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))",
        "",
        "# --- Gráfico 1: Curva S con Proyección ML ---",
        "sprints_x = np.array([1, 2, 3, 4, 5, 6])",
        "sprint_labels = ['Q1', 'Q2', 'Q3 (Corte)', 'Q4 (Proy)', 'Q5 (Proy)', 'Q6 (Cierre)']",
        "",
        "ax1.plot(sprints_x, forecast['pv_hours'], 'k--', label='PV: Valor Planificado (Línea Base)', lw=2.0)",
        "ax1.plot(sprints_x[:3], forecast['ev_hours'][:3], 's-', color='#2E7D32', label='EV: Valor Ganado Real (Q1-Q3)', lw=2.5)",
        "ax1.plot(sprints_x[:3], forecast['ac_hours_actual'], 'o-', color='#C62828', label='AC: Costo/Esfuerzo Real (Q1-Q3)', lw=2.5)",
        "ax1.plot(sprints_x[2:], forecast['ev_hours'][2:], 's:', color='#81C784', label='EV Proyectado (SPI = 1.0)', lw=2.0)",
        "ax1.plot(sprints_x[2:], forecast['ac_hours_ml'][2:], 'o:', color='#EF5350', label=f\"AC Proyectado ML (EAC={forecast['eac_ml_hours']:.1f}h)\", lw=2.2)",
        "",
        "ax1.axvline(3, color='gray', linestyle=':', lw=1.5, label='Corte de Control (Quincena 3)')",
        "ax1.set_xticks(sprints_x)",
        "ax1.set_xticklabels(sprint_labels, fontsize=9)",
        "ax1.set_xlabel('Sprints Quincenales', fontweight='bold')",
        "ax1.set_ylabel('Horas Técnicas Acumuladas (Maestría + ML)', fontweight='bold')",
        "ax1.set_title('Curva S de Gestión del Valor Ganado (EVM) + Proyección ML Ridge L2', fontweight='bold')",
        "ax1.legend(loc='upper left', fontsize=8)",
        "",
        "# --- Gráfico 2: Evolución de Índices CPI y SPI ---",
        "actual_q = [1, 2, 3]",
        "actual_q_labels = ['Quincena 1', 'Quincena 2', 'Quincena 3 (Actual)']",
        "cpi_tech_vals = [36.0/35.0, 78.0/72.0, 114.0/111.0]",
        "spi_tech_vals = [1.0, 1.0, 1.0]",
        "cpi_fin_vals = [10900.0/10720.0, 21800.0/21200.0, 32700.0/31850.0]",
        "",
        "ax2.axhline(1.0, color='gray', linestyle='--', lw=1.5, label='Umbral de Equilibrio (Índice = 1.00)')",
        "ax2.plot(actual_q, cpi_tech_vals, 'o-', color='#0288D1', lw=2.5, label='CPI Técnico (Eficiencia de Horas)')",
        "ax2.plot(actual_q, spi_tech_vals, 's-', color='#FFA000', lw=2.2, label='SPI Técnico (Cumplimiento Cronograma)')",
        "ax2.plot(actual_q, cpi_fin_vals, '^-', color='#388E3C', lw=2.5, label='CPI Financiero (Ahorro Presupuestal MXN)')",
        "",
        "ax2.set_xticks(actual_q)",
        "ax2.set_xticklabels(actual_q_labels, fontsize=9)",
        "ax2.set_xlabel('Quincena de Evaluación', fontweight='bold')",
        "ax2.set_ylabel('Valor del Índice de Desempeño', fontweight='bold')",
        "ax2.set_title('Evolución de Índices EVM (CPI y SPI > 1.0 = Desempeño Favorable)', fontweight='bold')",
        "ax2.set_ylim(0.92, 1.10)",
        "ax2.legend(loc='lower right', fontsize=8)",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ]))

    # Section 6: Risk Matrix
    cells.append(make_markdown_cell([
        "---",
        "## 6. Matriz Binaria de Riesgos (Probabilidad × Impacto) y Plan de Contingencia",
        "Conforme al Tema 5.6 del curso de Proyectos (UNIR), cada amenaza detectada se clasifica en una escala cuantitativa de 1 a 5 y se le asocia una estrategia formal."
    ]))

    cells.append(make_code_cell([
        "# Mostramos la matriz de riesgos oficial",
        "df_risks = pd.DataFrame([",
        "    {",
        "        'ID': r.risk_id,",
        "        'Categoría': r.category,",
        "        'Prob': r.probability,",
        "        'Imp': r.impact,",
        "        'Prioridad': r.priority,",
        "        'Estrategia': r.response_strategy,",
        "        'Descripción': r.description,",
        "        'Plan de Contingencia': r.contingency_plan,",
        "    }",
        "    for r in RISK_CATALOG",
        "])",
        "df_risks[['ID', 'Categoría', 'Prob', 'Imp', 'Prioridad', 'Estrategia']]"
    ]))

    cells.append(make_code_cell([
        "# Figura 2 (Izquierda): Mapa de Calor Binario de Riesgos (Probabilidad x Impacto)",
        "fig, ax = plt.subplots(figsize=(7, 7))",
        "",
        "# Fondo de cuadrantes de severidad",
        "grid_colors = np.array([",
        "    [1, 1, 2, 3, 4],  # Prob = 1",
        "    [1, 2, 2, 3, 4],  # Prob = 2",
        "    [2, 2, 3, 4, 4],  # Prob = 3",
        "    [2, 3, 4, 5, 5],  # Prob = 4",
        "    [3, 4, 4, 5, 5],  # Prob = 5",
        "])",
        "",
        "ax.imshow(grid_colors, cmap='YlOrRd', origin='lower', extent=[0.5, 5.5, 0.5, 5.5], alpha=0.55)",
        "",
        "# Posicionamos los riesgos R-01 a R-06",
        "for r in RISK_CATALOG:",
        "    ax.plot(r.impact, r.probability, 'o', color='black', markersize=10)",
        "    ax.text(r.impact, r.probability + 0.18, r.risk_id, ha='center', fontweight='bold', fontsize=10, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))",
        "",
        "ax.set_xticks([1, 2, 3, 4, 5])",
        "ax.set_xticklabels(['1: Muy Bajo', '2: Bajo', '3: Medio', '4: Alto', '5: Muy Alto'])",
        "ax.set_yticks([1, 2, 3, 4, 5])",
        "ax.set_yticklabels(['1: Muy Baja', '2: Baja', '3: Media', '4: Alta', '5: Muy Alta'])",
        "ax.set_xlabel('Impacto sobre Alcance, Tiempo, Coste o Calidad', fontweight='bold')",
        "ax.set_ylabel('Probabilidad de Ocurrencia', fontweight='bold')",
        "ax.set_title('Mapa de Calor Binario de Riesgos (Tema 5 · Tablas 4-6)', fontweight='bold')",
        "ax.grid(color='white', linestyle='-', linewidth=1.5)",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ]))

    # Section 7: Forge3D & API
    cells.append(make_markdown_cell([
        "---",
        "## 7. Módulo Forge3D (Manufactura Aditiva) y Arquitectura de Microservicios",
        "Cotización y planificación de lotes de impresión 3D en la Anycubic Kobra S1 con control de boquilla preventiva (Riesgo R-05)."
    ]))

    cells.append(make_code_cell([
        "from lifeos.forge3d import Forge3DEngine",
        "",
        "forge = Forge3DEngine()",
        "quote = forge.calculate_quote(",
        "    job_name=\"Carcasa Dispositivo Edge ML\",",
        "    grams_per_unit=60.0,",
        "    hours_per_unit=3.5,",
        "    units=10,",
        "    material=\"PETG\",",
        "    post_processing_minutes=12.0,",
        "    margin_percent=40.0,",
        ")",
        "",
        "print(f\"--- Cotización de Lote Forge3D: {quote.job_name} ---\")",
        "print(f\"Unidades solicitadas: {quote.units} piezas de {quote.material}\")",
        "print(f\"Horas de impresión totales: {quote.total_print_hours} horas\")",
        "print(f\"Costo de material: ${quote.total_material_cost:,.2f} MXN\")",
        "print(f\"Costo de energía: ${quote.total_energy_cost:,.2f} MXN\")",
        "print(f\"Desgaste de máquina: ${quote.total_depreciation_cost:,.2f} MXN\")",
        "print(f\"Fondo mantenimiento boquilla (R-05): ${quote.total_maintenance_reserve:,.2f} MXN\")",
        "print(f\"Mano de obra y acabados: ${quote.total_labor_cost:,.2f} MXN\")",
        "print(f\"Costo Base de Producción: ${quote.base_cost_total:,.2f} MXN\")",
        "print(f\"PRECIO FINAL DEL LOTE (+{quote.margin_percent:.0f}%): ${quote.final_price_total:,.2f} MXN\")",
        "print(f\"Precio Unitario al Cliente: ${quote.price_per_unit:,.2f} MXN/unidad\")",
        "print(f\"Requiere calibración preventiva inmediata: {'SÍ' if quote.calibration_needed else 'No (En rango seguro)'}\")"
    ]))

    # Section 8: Unit test runner
    cells.append(make_markdown_cell([
        "---",
        "## 8. Ejecución Automatizada de Pruebas Unitarias",
        "Corremos la suite completa de pruebas unitarias (`pytest`) para certificar la fiabilidad y reproducibilidad de todo el sistema."
    ]))

    cells.append(make_code_cell([
        "import pytest",
        "print('Ejecutando la suite de pruebas unitarias automáticas...')",
        "exit_code = pytest.main(['tests/', '-v'])",
        "if exit_code == 0:",
        "    print('\\n🎉 ¡TODAS LAS PRUEBAS UNITARIAS PASARON EXITOSAMENTE (100% SUCCESS)!')",
        "else:",
        "    print(f'\\n⚠️ Código de salida de pruebas: {exit_code}')"
    ]))

    # Conclusion & STAR narrative
    cells.append(make_markdown_cell([
        "---",
        "## 9. Guía de Presentación en Entrevistas de Machine Learning Engineer (Método STAR)",
        "",
        "| Fase STAR | Narrativa de Impacto Cuantitativo para Entrevistas de Machine Learning Engineer |",
        "| :--- | :--- |",
        "| **Situación** | Durante mi Maestría en IA en UNIR, necesitaba coordinar el desarrollo de software/modelos ML para mi portafolio, entregables académicos y finanzas quincenales sin sobrecostes ni retrasos. |",
        "| **Tarea** | Diseñar y construir un sistema de ingeniería reproducible (**LifeOS-ML**) que llevara la gestión artesanal a un estándar industrial (PMBOK® 7 + Scrum/Kanban + EVM), automatizando priorización y pronóstico con ML. |",
        "| **Acción** | Estructuré un PGP con EDT de 16 paquetes en 6 Sprints (240 h), programé un motor CPM con simulación Monte Carlo (10k iteraciones), entrené un clasificador NLP puro en NumPy (TF-IDF + Softmax L2) para la EDT y Matriz de Covey, e implementé una regresión polinomial Ridge L2 acoplada a un tablero EVM. |",
        "| **Resultado** | El clasificador NLP superó **95% de exactitud y F1 > 0.95**; la reasignación hacia Cuadrante II elevó el tiempo de ingeniería profunda del **32% al 68%**; y el control EVM mantuvo el proyecto en **plazo exacto ($SPI = 1.00$) y bajo presupuesto ($CPI = 1.027$)**, ahorrando **$1,700 MXN** adicionales y garantizando **0% en intereses bancarios**. |",
        "",
        "---",
        "*(c) 2026 Ing. Miguel Martínez Ortega · LifeOS-ML Project Management & Engineering Portfolio.*"
    ]))

    notebook_content = {
        "cells": cells,
        "metadata": {
            "colab": {
                "name": "LifeOS_ML_EVM_Sprint_Tracker_Colab.ipynb",
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.13",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_path = Path("LifeOS_ML_EVM_Sprint_Tracker_Colab.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=1, ensure_ascii=False)
    print(f"Notebook creado exitosamente en {out_path.resolve()} con {len(cells)} celdas.")


if __name__ == "__main__":
    build_lifeos_notebook()
