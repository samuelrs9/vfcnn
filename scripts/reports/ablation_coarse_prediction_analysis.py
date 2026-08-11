"""
Ablation Study: Coarse Prediction Impact Analysis
==================================================

📖 DOCUMENTAÇÃO COMPLETA: Ver ABLATION_COARSE_PREDICTION_README.md

OBJETIVO:
    Realizar estudo de ablação comparando o desempenho da rede neural esparsa
    com e sem a etapa de "coarse prediction" (predição grosseira).

DESCRIÇÃO:
    Este script analisa dados de inferência de uma rede neural esparsa para
    simulações de fluidos, comparando duas estratégias:
    
    1. WITH COARSE: Processo de 2 estágios
       - Coarse prediction: predição grosseira para identificar região de interesse
       - Fine prediction: predição refinada apenas na região identificada
    
    2. WITHOUT COARSE: Processo de 1 estágio
       - Fine prediction direta em toda a região
    
    O estudo avalia o trade-off entre tempo de inferência e métricas de acurácia
    (F1-Score, Combined Metric, Precision, Recall, Matthews Coefficient).

ENTRADA:
    - CSV com resultados de predições: predictions_comparison.csv
      Deve conter colunas:
        * simulation: nome da simulação
        * prediction: nome do modelo (com sufixo _no_coarse para sem coarse)
        * recall, precision, f1_score, combined_metric, matthews_coeff
        * coarse_prediction_time, fine_prediction_time, total_time

SAÍDA:
    Diretório: reports/ablation_coarse_prediction/
    
    Tabelas CSV:
        - coarse_prediction_comparison.csv: comparação detalhada
        - coarse_prediction_summary.csv: resumo simplificado
    
    Visualizações PNG (7 gráficos):
        - time_vs_f1_scatter.png: scatter plot tempo vs F1-Score
        - time_vs_combined_metric_scatter.png: scatter plot tempo vs Combined Metric
        - tradeoff_analysis.png: análise completa de trade-offs
        - speedup_factor.png: fator de aceleração por simulação
        - f1_difference.png: impacto na acurácia (F1)
        - time_comparison_bars.png: barras comparando tempos
        - time_breakdown.png: decomposição do tempo (2 vs 1 estágio)

USO:
    python ablation_coarse_prediction_analysis.py
    
    ou com ambiente conda:
    conda run -n vfnet python ablation_coarse_prediction_analysis.py

AUTOR: VoxelFluidNet Team
DATA: 2026-06-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# CONFIGURAÇÃO DE VISUALIZAÇÃO
# ============================================================================
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# CARREGAMENTO DE DADOS
# ============================================================================
csv_path = "/work1/voxel-fluid-net/reports/predictions_comparison_report/predictions_comparison.csv"
df = pd.read_csv(csv_path)

print("=" * 80)
print("ABLATION STUDY: COARSE PREDICTION IMPACT")
print("=" * 80)
print(f"\nTotal rows: {len(df)}")
print(f"Unique simulations: {df['simulation'].nunique()}")

# ============================================================================
# IDENTIFICAÇÃO E CLASSIFICAÇÃO DOS MODELOS
# ============================================================================
# Identifica predições com e sem coarse prediction baseado no sufixo '_no_coarse'
df['has_coarse'] = ~df['prediction'].str.contains('no_coarse')
df['model_type'] = df['has_coarse'].map({True: 'With Coarse', False: 'Without Coarse'})

# Extrai nome base do modelo para pareamento (remove sufixo _no_coarse)
df['base_model'] = df['prediction'].str.replace('_no_coarse', '')

# ============================================================================
# CONSTRUÇÃO DA TABELA COMPARATIVA ESTRUTURADA
# ============================================================================
# Para cada simulação, pareia os resultados com e sem coarse prediction
# e calcula as diferenças em tempo e métricas de acurácia
results = []

for sim in df['simulation'].unique():
    sim_data = df[df['simulation'] == sim]
    
    # Separa resultados com e sem coarse prediction para a mesma simulação
    with_coarse = sim_data[sim_data['has_coarse']]
    without_coarse = sim_data[~sim_data['has_coarse']]
    
    # Processa apenas se ambas as versões existem para comparação
    if len(with_coarse) > 0 and len(without_coarse) > 0:
        wc = with_coarse.iloc[0]  # with coarse
        woc = without_coarse.iloc[0]  # without coarse
        
        # Dicionário com todas as métricas para comparação
        result = {
            'simulation': sim,
            
            # ==== MÉTRICAS COM COARSE PREDICTION ====
            'recall_with_coarse': wc['recall'],
            'precision_with_coarse': wc['precision'],
            'f1_with_coarse': wc['f1_score'],
            'combined_metric_with_coarse': wc['combined_metric'],
            'matthews_with_coarse': wc['matthews_coeff'],
            
            # ==== TEMPO COM COARSE PREDICTION (2 estágios) ====
            'coarse_time': wc['coarse_prediction_time'],
            'fine_time_with_coarse': wc['fine_prediction_time'],
            'total_time_with_coarse': wc['total_time'],
            
            # ==== MÉTRICAS SEM COARSE PREDICTION ====
            'recall_without_coarse': woc['recall'],
            'precision_without_coarse': woc['precision'],
            'f1_without_coarse': woc['f1_score'],
            'combined_metric_without_coarse': woc['combined_metric'],
            'matthews_without_coarse': woc['matthews_coeff'],
            
            # ==== TEMPO SEM COARSE PREDICTION (1 estágio) ====
            'fine_time_without_coarse': woc['fine_prediction_time'],
            'total_time_without_coarse': woc['total_time'],
            
            # ==== DIFERENÇAS CALCULADAS ====
            # Tempo: valor negativo indica que WITHOUT é mais rápido
            'time_diff_absolute': woc['total_time'] - wc['total_time'],
            'time_diff_percentage': ((woc['total_time'] - wc['total_time']) / wc['total_time']) * 100,
            
            # Acurácia: valor positivo indica que WITHOUT é melhor
            'recall_diff': woc['recall'] - wc['recall'],
            'precision_diff': woc['precision'] - wc['precision'],
            'f1_diff': woc['f1_score'] - wc['f1_score'],
            'combined_metric_diff': woc['combined_metric'] - wc['combined_metric'],
            'matthews_diff': woc['matthews_coeff'] - wc['matthews_coeff'],
            
            # Fator de aceleração: valores > 1 indicam que WITH é mais rápido
            'speedup_factor': wc['total_time'] / woc['total_time'],
        }
        results.append(result)

# Cria DataFrame com todas as comparações
comparison_df = pd.DataFrame(results)

# ============================================================================
# SALVAMENTO DAS TABELAS REESTRUTURADAS
# ============================================================================
output_dir = Path("reports/ablation_coarse_prediction")
output_dir.mkdir(parents=True, exist_ok=True)

# Tabela completa com todas as métricas e diferenças
comparison_df.to_csv(output_dir / "coarse_prediction_comparison.csv", index=False)

# Tabela resumida com apenas as métricas principais
summary_cols = [
    'simulation',
    'total_time_with_coarse',
    'total_time_without_coarse',
    'time_diff_percentage',
    'speedup_factor',
    'f1_with_coarse',
    'f1_without_coarse',
    'f1_diff',
    'combined_metric_with_coarse',
    'combined_metric_without_coarse',
    'combined_metric_diff'
]
summary_df = comparison_df[summary_cols].copy()
summary_df.to_csv(output_dir / "coarse_prediction_summary.csv", index=False)

# ============================================================================
# ESTATÍSTICAS RESUMIDAS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"\nNumber of simulations compared: {len(comparison_df)}")
print(f"\nTime Analysis:")
print(f"  Average total time WITH coarse: {comparison_df['total_time_with_coarse'].mean():.3f}s")
print(f"  Average total time WITHOUT coarse: {comparison_df['total_time_without_coarse'].mean():.3f}s")
print(f"  Average time difference: {comparison_df['time_diff_absolute'].mean():.3f}s ({comparison_df['time_diff_percentage'].mean():.1f}%)")
print(f"  Average speedup factor: {comparison_df['speedup_factor'].mean():.3f}x")
print(f"\nAccuracy Analysis (F1-Score):")
print(f"  Average F1 WITH coarse: {comparison_df['f1_with_coarse'].mean():.4f}")
print(f"  Average F1 WITHOUT coarse: {comparison_df['f1_without_coarse'].mean():.4f}")
print(f"  Average F1 difference: {comparison_df['f1_diff'].mean():.4f}")
print(f"\nAccuracy Analysis (Combined Metric):")
print(f"  Average Combined Metric WITH coarse: {comparison_df['combined_metric_with_coarse'].mean():.4f}")
print(f"  Average Combined Metric WITHOUT coarse: {comparison_df['combined_metric_without_coarse'].mean():.4f}")
print(f"  Average Combined Metric difference: {comparison_df['combined_metric_diff'].mean():.4f}")

# Tabela comparativa detalhada por simulação
print("\n" + "=" * 80)
print("DETAILED COMPARISON BY SIMULATION")
print("=" * 80)
print(summary_df.to_string(index=False))

# ============================================================================
# VISUALIZAÇÕES
# ============================================================================
# Geração de 7 gráficos diferentes para análise visual do trade-off
# entre tempo de inferência e acurácia

# ============================================================================
# GRÁFICO 1: Scatter Plot - Tempo vs F1-Score
# ============================================================================
# Mostra a relação entre tempo de inferência e F1-Score
# Setas conectam os pontos da mesma simulação (com -> sem coarse)
fig, ax = plt.subplots(figsize=(12, 8))

# Plot points with coarse
ax.scatter(comparison_df['total_time_with_coarse'], 
           comparison_df['f1_with_coarse'],
           s=200, alpha=0.7, c='#2E86AB', marker='o', 
           label='With Coarse Prediction', edgecolors='black', linewidth=1.5)

# Plot points without coarse
ax.scatter(comparison_df['total_time_without_coarse'], 
           comparison_df['f1_without_coarse'],
           s=200, alpha=0.7, c='#A23B72', marker='s', 
           label='Without Coarse Prediction', edgecolors='black', linewidth=1.5)

# Draw arrows showing the transition
for idx, row in comparison_df.iterrows():
    ax.annotate('', 
                xy=(row['total_time_without_coarse'], row['f1_without_coarse']),
                xytext=(row['total_time_with_coarse'], row['f1_with_coarse']),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.5))

ax.set_xlabel('Total Inference Time (seconds)', fontsize=14, fontweight='bold')
ax.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
ax.set_title('Inference Time vs F1-Score: Impact of Coarse Prediction', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper right', bbox_to_anchor=(0.98, 0.98))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "time_vs_f1_scatter.png", dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_dir / 'time_vs_f1_scatter.png'}")

# ============================================================================
# GRÁFICO 2: Scatter Plot - Tempo vs Combined Metric
# ============================================================================
# Similar ao gráfico 1, mas usando Combined Metric ao invés de F1
fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(comparison_df['total_time_with_coarse'], 
           comparison_df['combined_metric_with_coarse'],
           s=200, alpha=0.7, c='#2E86AB', marker='o', 
           label='With Coarse Prediction', edgecolors='black', linewidth=1.5)

ax.scatter(comparison_df['total_time_without_coarse'], 
           comparison_df['combined_metric_without_coarse'],
           s=200, alpha=0.7, c='#A23B72', marker='s', 
           label='Without Coarse Prediction', edgecolors='black', linewidth=1.5)

for idx, row in comparison_df.iterrows():
    ax.annotate('', 
                xy=(row['total_time_without_coarse'], row['combined_metric_without_coarse']),
                xytext=(row['total_time_with_coarse'], row['combined_metric_with_coarse']),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.5))

ax.set_xlabel('Total Inference Time (seconds)', fontsize=14, fontweight='bold')
ax.set_ylabel('Combined Metric', fontsize=14, fontweight='bold')
ax.set_title('Inference Time vs Combined Metric: Impact of Coarse Prediction', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper right', bbox_to_anchor=(0.98, 0.98))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "time_vs_combined_metric_scatter.png", dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'time_vs_combined_metric_scatter.png'}")

# ============================================================================
# GRÁFICO 3: Barras - Comparação de Tempo por Simulação
# ============================================================================
# Barras lado a lado comparando tempo total com e sem coarse prediction
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(comparison_df))
width = 0.35

bars1 = ax.bar(x - width/2, comparison_df['total_time_with_coarse'], 
               width, label='With Coarse', color='#2E86AB', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, comparison_df['total_time_without_coarse'], 
               width, label='Without Coarse', color='#A23B72', alpha=0.8, edgecolor='black')

ax.set_xlabel('Simulation', fontsize=14, fontweight='bold')
ax.set_ylabel('Total Inference Time (seconds)', fontsize=14, fontweight='bold')
ax.set_title('Inference Time Comparison by Simulation', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['simulation'], rotation=45, ha='right')
ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0.02, 0.98))
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "time_comparison_bars.png", dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'time_comparison_bars.png'}")

# ============================================================================
# GRÁFICO 4: Fator de Aceleração (Speedup)
# ============================================================================
# Mostra quantas vezes o método WITH coarse é mais rápido que WITHOUT
# Valores < 1 (vermelho) = WITHOUT é mais rápido
# Valores > 1 (verde) = WITH é mais rápido
fig, ax = plt.subplots(figsize=(14, 8))

colors = ['#E63946' if x < 1 else '#06D6A0' for x in comparison_df['speedup_factor']]
bars = ax.bar(comparison_df['simulation'], comparison_df['speedup_factor'], 
              color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.axhline(y=1, color='black', linestyle='--', linewidth=2, label='No change')
ax.set_xlabel('Simulation', fontsize=14, fontweight='bold')
ax.set_ylabel('Speedup Factor (With Coarse / Without Coarse)', fontsize=14, fontweight='bold')
ax.set_title('Speedup Factor by Using Coarse Prediction', fontsize=16, fontweight='bold', pad=20)
ax.set_xticklabels(comparison_df['simulation'], rotation=45, ha='right')
ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0.02, 0.98))
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}x',
            ha='center', va='bottom' if height > 1 else 'top', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "speedup_factor.png", dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'speedup_factor.png'}")

# ============================================================================
# GRÁFICO 5: Diferença de Acurácia (F1-Score)
# ============================================================================
# Mostra o impacto na acurácia ao remover coarse prediction
# Valores positivos (verde) = melhora sem coarse
# Valores negativos (vermelho) = piora sem coarse
fig, ax = plt.subplots(figsize=(14, 8))

colors = ['#E63946' if x < 0 else '#06D6A0' for x in comparison_df['f1_diff']]
bars = ax.bar(comparison_df['simulation'], comparison_df['f1_diff'], 
              color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('Simulation', fontsize=14, fontweight='bold')
ax.set_ylabel('F1-Score Difference (Without - With Coarse)', fontsize=14, fontweight='bold')
ax.set_title('F1-Score Impact of Removing Coarse Prediction', fontsize=16, fontweight='bold', pad=20)
ax.set_xticklabels(comparison_df['simulation'], rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:+.4f}',
            ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "f1_difference.png", dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'f1_difference.png'}")

# ============================================================================
# GRÁFICO 6: Decomposição de Tempo (2 estágios vs 1 estágio)
# ============================================================================
# Compara a composição do tempo:
# - COM coarse: tempo coarse (laranja) + tempo fine (azul)
# - SEM coarse: apenas tempo fine (roxo)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# With coarse prediction
x = np.arange(len(comparison_df))
ax1.bar(x, comparison_df['coarse_time'], label='Coarse Prediction', 
        color='#F18F01', alpha=0.8, edgecolor='black')
ax1.bar(x, comparison_df['fine_time_with_coarse'], 
        bottom=comparison_df['coarse_time'], 
        label='Fine Prediction', color='#2E86AB', alpha=0.8, edgecolor='black')
ax1.set_xlabel('Simulation', fontsize=12, fontweight='bold')
ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_title('With Coarse Prediction\n(Two-Stage Process)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(comparison_df['simulation'], rotation=45, ha='right', fontsize=9)
ax1.legend(fontsize=11, loc='upper left', bbox_to_anchor=(0.02, 0.98))
ax1.grid(True, alpha=0.3, axis='y')

# Without coarse prediction
ax2.bar(x, comparison_df['fine_time_without_coarse'], 
        label='Fine Prediction Only', color='#A23B72', alpha=0.8, edgecolor='black')
ax2.set_xlabel('Simulation', fontsize=12, fontweight='bold')
ax2.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
ax2.set_title('Without Coarse Prediction\n(Single-Stage Process)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(comparison_df['simulation'], rotation=45, ha='right', fontsize=9)
ax2.legend(fontsize=11, loc='upper left', bbox_to_anchor=(0.02, 0.98))
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / "time_breakdown.png", dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'time_breakdown.png'}")

# ============================================================================
# GRÁFICO 7: Análise de Trade-off (Tempo Ganho vs Acurácia Perdida)
# ============================================================================
# Gráfico de quadrantes mostrando:
# - Eixo X: economia de tempo (%)
# - Eixo Y: mudança no F1-Score (%)
# - Cor: fator de speedup
# Quadrantes:
#   - Superior esquerdo: Mais rápido E melhor (ideal)
#   - Inferior esquerdo: Mais rápido MAS pior (trade-off aceitável)
#   - Superior direito: Mais lento MAS melhor (indesejável)
#   - Inferior direito: Mais lento E pior (pior caso)
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot showing the trade-off
scatter = ax.scatter(comparison_df['time_diff_percentage'], 
                    comparison_df['f1_diff'] * 100,  # Convert to percentage
                    s=300, alpha=0.7, c=comparison_df['speedup_factor'], 
                    cmap='RdYlGn', edgecolors='black', linewidth=2)

# Add simulation labels
for idx, row in comparison_df.iterrows():
    ax.annotate(row['simulation'].replace('_3d_big_res', ''), 
                xy=(row['time_diff_percentage'], row['f1_diff'] * 100),
                xytext=(5, 5), textcoords='offset points', 
                fontsize=9, alpha=0.7)

# Add quadrant lines
ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Speedup Factor', fontsize=12, fontweight='bold')

ax.set_xlabel('Time Saved (% reduction)', fontsize=14, fontweight='bold')
ax.set_ylabel('F1-Score Change (%)', fontsize=14, fontweight='bold')
ax.set_title('Trade-off: Time Savings vs Accuracy Impact\n(Without Coarse vs With Coarse)', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)

# Add text annotations for quadrants
ax.text(0.95, 0.95, 'Slower & Better\n(Undesirable)', 
        transform=ax.transAxes, ha='right', va='top', 
        fontsize=10, style='italic', alpha=0.5,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
ax.text(0.05, 0.95, 'Faster & Better\n(Ideal)', 
        transform=ax.transAxes, ha='left', va='top', 
        fontsize=10, style='italic', alpha=0.5,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
ax.text(0.95, 0.05, 'Slower & Worse\n(Worst)', 
        transform=ax.transAxes, ha='right', va='bottom', 
        fontsize=10, style='italic', alpha=0.5,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
ax.text(0.05, 0.05, 'Faster & Worse\n(Trade-off)', 
        transform=ax.transAxes, ha='left', va='bottom', 
        fontsize=10, style='italic', alpha=0.5,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(output_dir / "tradeoff_analysis.png", dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'tradeoff_analysis.png'}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nAll results saved to: {output_dir.absolute()}")
print(f"\nGenerated files:")
print(f"  - coarse_prediction_comparison.csv (detailed comparison)")
print(f"  - coarse_prediction_summary.csv (simplified summary)")
print(f"  - 7 visualization plots (.png)")
print("\n" + "=" * 80)
