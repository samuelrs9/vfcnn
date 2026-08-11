"""
Script para adicionar estimativas de tempo sem coarse prediction

Para predições WITH coarse, estima o tempo que seria necessário se todas as partículas
(100%) tivessem sido processadas na etapa de fine prediction, ao invés de apenas
as partículas undefined (undefined_percentage).

Fórmula: 
    fine_prediction_estimated = fine_prediction_time / (undefined_percentage / 100)
    total_time_estimated = coarse_time=0 + fine_prediction_estimated
"""

import pandas as pd
import os

# Caminhos dos arquivos
predictions_file = '/work1/voxel-fluid-net/reports/predictions_comparison_report/predictions_comparison.csv'
coarse_stats_file = '/work1/voxel-fluid-net/reports/coarse_prediction_comparison/coarse_prediction_comparison.csv'
output_file = '/work1/voxel-fluid-net/reports/predictions_comparison_report/predictions_comparison_with_estimates.csv'

# Carregar dados
print("Carregando dados...")
df_predictions = pd.read_csv(predictions_file)
df_coarse_stats = pd.read_csv(coarse_stats_file)

# Criar mapeamento simulação -> undefined_percentage
undefined_map = {}
for _, row in df_coarse_stats.iterrows():
    sim_name = row['simulation']
    undefined_pct = row['undefined_percentage']
    undefined_map[sim_name] = undefined_pct

print(f"\nUndefined percentages por simulação:")
for sim, pct in undefined_map.items():
    print(f"  {sim}: {pct:.2f}%")

# Lista para armazenar novas linhas estimadas
new_rows = []

# Processar cada predição
print(f"\nProcessando predições...")
for idx, row in df_predictions.iterrows():
    simulation = row['simulation']
    prediction_type = row['prediction_type']
    coarse_usage = row['coarse_usage']
    
    # Apenas processar predições REGIONWISE (não sparse_regionwise) com WITH coarse
    # Sparse_regionwise já tem versões no_coarse reais, não precisa de estimativas
    if prediction_type == 'regionwise' and coarse_usage == 'with_coarse' and simulation in undefined_map:
        undefined_pct = undefined_map[simulation]
        
        # Calcular tempo estimado se 100% das partículas fossem processadas
        fine_time_original = row['fine_prediction_time']
        
        # Fórmula: tempo_total = tempo / (percentual_processado / 100)
        # Se processou X% das partículas em T segundos, 
        # processar 100% levaria: T / (X/100)
        fine_time_estimated = fine_time_original / (undefined_pct / 100.0)
        total_time_estimated = 0.0 + fine_time_estimated  # sem coarse
        
        # Criar nova linha com a estimativa
        new_row = row.copy()
        new_row['prediction'] = row['prediction'] + '_no_coarse_estimated'
        new_row['coarse_usage'] = 'no_coarse_estimated'
        new_row['coarse_prediction_time'] = 0.0
        new_row['fine_prediction_time'] = fine_time_estimated
        new_row['total_time'] = total_time_estimated
        # Manter imagens generation em 0 (já incluído no fine_time para regionwise)
        
        new_rows.append(new_row)
        
        print(f"\n  {simulation} - {row['prediction']}")
        print(f"    Undefined: {undefined_pct:.2f}%")
        print(f"    Fine time original: {fine_time_original:.2f}s")
        print(f"    Fine time estimado (100%): {fine_time_estimated:.2f}s")
        print(f"    Total time estimado: {total_time_estimated:.2f}s")

# Combinar dados originais com estimativas
print(f"\n{'='*80}")
print(f"Adicionando {len(new_rows)} linhas estimadas...")
df_combined = pd.concat([df_predictions, pd.DataFrame(new_rows)], ignore_index=True)

# Ordenar por simulação e nome de predição
df_combined = df_combined.sort_values(['simulation', 'prediction'])

# Salvar resultado
df_combined.to_csv(output_file, index=False)

print(f"\n{'='*80}")
print(f"Arquivo salvo: {output_file}")
print(f"Total de linhas: {len(df_combined)} ({len(df_predictions)} originais + {len(new_rows)} estimadas)")
print(f"{'='*80}")

# Mostrar resumo por simulação
print("\nResumo por simulação:")
for sim in df_combined['simulation'].unique():
    sim_data = df_combined[df_combined['simulation'] == sim]
    print(f"\n  {sim}:")
    print(f"    Total predições: {len(sim_data)}")
    print(f"    With coarse: {len(sim_data[sim_data['coarse_usage'] == 'with_coarse'])}")
    print(f"    No coarse: {len(sim_data[sim_data['coarse_usage'] == 'no_coarse'])}")
    print(f"    No coarse (estimado): {len(sim_data[sim_data['coarse_usage'] == 'no_coarse_estimated'])}")
