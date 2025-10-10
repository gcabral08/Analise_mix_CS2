import streamlit as st
import pandas as pd
import json
import numpy as np # Adicionado para cálculos
from datetime import datetime, date
import matplotlib.pyplot as plt
import io

# --- LÓGICA DE ANÁLISE (CLASSE StatsAnalyzer) ---
class StatsAnalyzer:
    def __init__(self, data):
        if not data:
            raise ValueError("Os dados das partidas não foram carregados.")
        self.raw_data = data
        self.df = self._load_data_into_dataframe()

    def _unify_player_names(self, name):
        name_map = {'Birigul': 'Birigu1', 'Craque Perfil Líder': 'Craque Perfil Lider', 'F1dellis': 'Fidellis', 'BOSSS_': 'BOSSS', 'FzrG': 'Fzr'}
        return name_map.get(name, name)

    def _load_data_into_dataframe(self):
        records = []
        for match_id, match in enumerate(self.raw_data):
            score_a = match.get('score_a', 0)
            score_b = match.get('score_b', 0)
            
            is_team_a_winner = None
            if score_a > score_b: is_team_a_winner = True
            elif score_b > score_a: is_team_a_winner = False
            
            team_a_players = {self._unify_player_names(p.get('player')) for p in match.get('team_a', [])}
            team_b_players = {self._unify_player_names(p.get('player')) for p in match.get('team_b', [])}

            for player_stats in match.get('team_a', []):
                player_name = self._unify_player_names(player_stats.get('player'))
                records.append({'match_id': match_id, 'date': match.get('date'), 'map': match.get('map'),'player': player_name, 'k': player_stats.get('k', 0), 'a': player_stats.get('a', 0), 'd': player_stats.get('d', 0),'team_roster': frozenset(team_a_players), 'won': is_team_a_winner, 'round_diff': score_a - score_b,'rounds_ganhos': score_a, 'rounds_perdidos': score_b,'teammates': [self._unify_player_names(p.get('player')) for p in match.get('team_a', []) if p.get('player') != player_stats.get('player')],'opponents': [self._unify_player_names(p.get('player')) for p in match.get('team_b', [])]})
            
            for player_stats in match.get('team_b', []):
                player_name = self._unify_player_names(player_stats.get('player'))
                records.append({'match_id': match_id, 'date': match.get('date'), 'map': match.get('map'),'player': player_name, 'k': player_stats.get('k', 0), 'a': player_stats.get('a', 0), 'd': player_stats.get('d', 0),'team_roster': frozenset(team_b_players), 'won': is_team_a_winner is not None and not is_team_a_winner, 'round_diff': score_b - score_a,'rounds_ganhos': score_b, 'rounds_perdidos': score_a,'teammates': [self._unify_player_names(p.get('player')) for p in match.get('team_b', []) if p.get('player') != player_stats.get('player')],'opponents': [self._unify_player_names(p.get('player')) for p in match.get('team_a', [])]})
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def _filter_by_date(self, start_date, end_date):
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())
        return self.df[(self.df['date'] >= start_dt) & (self.df['date'] <= end_dt)]
    
    def get_player_list(self): return sorted(self.df['player'].unique())
    def get_map_list(self): return sorted(self.df['map'].unique())
    
    def get_player_cumulative_trend(self, player_name, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        player_df = df[df['player'] == player_name].sort_values(by='date').copy()
        if player_df.empty: return pd.DataFrame()
        match_stats = player_df.groupby('match_id').agg(date=('date', 'first'), k=('k', 'first'), d=('d', 'first'), won=('won', 'first')).sort_values(by='date').reset_index()
        match_stats['K_Cumulativo'] = match_stats['k'].cumsum()
        match_stats['D_Cumulativo'] = match_stats['d'].cumsum()
        match_stats['Taxa K/D Cumulativa'] = (match_stats['K_Cumulativo'] / match_stats['D_Cumulativo']).fillna(0)
        match_stats['Vitórias_Cumulativas'] = match_stats['won'].cumsum()
        match_stats['Partidas_Totais'] = range(1, len(match_stats) + 1)
        match_stats['% de Vitória Cumulativa'] = (match_stats['Vitórias_Cumulativas'] / match_stats['Partidas_Totais']) * 100
        return match_stats[['date', 'match_id', 'Taxa K/D Cumulativa', '% de Vitória Cumulativa']]

    def get_overall_player_stats(self, start_date, end_date, sort_by='Partidas Jogadas'):
        df = self._filter_by_date(start_date, end_date)
        if df.empty: return pd.DataFrame()
        stats = df.groupby('player').agg(**{'Partidas Jogadas': ('match_id', 'nunique'),'Abates (K)': ('k', 'sum'),'Assistências (A)': ('a', 'sum'),'Mortes (D)': ('d', 'sum'), 'Saldo de Rounds': ('round_diff', 'sum')})
        wins = df[df['won'] == True].groupby('player')['won'].count()
        stats['Vitórias'] = wins.reindex(stats.index, fill_value=0)
        stats['Derrotas'] = stats['Partidas Jogadas'] - stats['Vitórias']
        stats['% de Vitória'] = (stats['Vitórias'] / stats['Partidas Jogadas']).fillna(0) * 100
        stats['Taxa K/D'] = (stats['Abates (K)'] / stats['Mortes (D)']).replace([np.inf, -np.inf], 0).fillna(0)
        stats = stats[['Partidas Jogadas', 'Vitórias', 'Derrotas', '% de Vitória', 'Abates (K)', 'Assistências (A)', 'Mortes (D)', 'Taxa K/D', 'Saldo de Rounds']]
        return stats.sort_values(by=sort_by, ascending=False)

    def get_map_leaderboard(self, map_name, start_date, end_date, sort_by='% de Vitória'):
        # Reutiliza a função geral e filtra pelo mapa
        df = self._filter_by_date(start_date, end_date)
        map_df = df[df['map'] == map_name]
        if map_df.empty: return pd.DataFrame()
        
        # Recria o analyzer com dados filtrados para usar o método geral
        temp_analyzer = StatsAnalyzer([match for match_id, match in self.raw_data if match.get('map') == map_name])
        return temp_analyzer.get_overall_player_stats(start_date, end_date, sort_by)

    def get_player_overall_stats_summary(self, player_name, start_date, end_date):
        # Usa o método geral para pegar os dados do jogador
        player_stats = self.get_overall_player_stats(start_date, end_date)
        if player_name not in player_stats.index: return None
        
        summary = player_stats.loc[player_name]
        return {
            "Partidas Jogadas": int(summary['Partidas Jogadas']), 
            "Vitórias": int(summary['Vitórias']), 
            "Derrotas": int(summary['Derrotas']), 
            "% de Vitória": f"{summary['% de Vitória']:.2f}%",
            "Saldo de Rounds": int(summary['Saldo de Rounds'])
        }

    def get_performance_by_map(self, player_name, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        player_df = df[df['player'] == player_name]
        if player_df.empty: return pd.DataFrame()
        stats_by_map = player_df.groupby('map').agg(Partidas=('match_id', 'nunique'), Abates=('k', 'sum'), Mortes=('d', 'sum'), Saldo_Rounds=('round_diff', 'sum'))
        wins_by_map = player_df[player_df['won']].groupby('map')['match_id'].nunique()
        stats_by_map['Vitórias'] = wins_by_map.reindex(stats_by_map.index, fill_value=0)
        stats_by_map['% de Vitória'] = (stats_by_map['Vitórias'] / stats_by_map['Partidas']) * 100
        stats_by_map['K/D'] = (stats_by_map['Abates'] / stats_by_map['Mortes']).replace([np.inf, -np.inf], 0).fillna(0)
        return stats_by_map[['Partidas', 'Vitórias', '% de Vitória', 'K/D', 'Saldo_Rounds']].sort_values(by='Partidas', ascending=False)
    
    def get_h2h_overall(self, player1, player2, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        p1_vs_p2_matches = df[df['player'] == player1]['opponents'].apply(lambda x: player2 in x)
        match_ids = df[df['player'] == player1][p1_vs_p2_matches]['match_id'].unique()
        if len(match_ids) == 0: return None
        
        h2h_df = df[df['match_id'].isin(match_ids)]
        p1_stats = h2h_df[h2h_df['player'] == player1]
        p2_stats = h2h_df[h2h_df['player'] == player2]
        
        p1_wins = p1_stats.groupby('match_id')['won'].first().sum()
        total_games = len(match_ids)
        
        return {
            "Partidas H2H": total_games,
            f"Vitórias {player1}": int(p1_wins),
            f"Vitórias {player2}": total_games - int(p1_wins),
            f"Kills {player1}": p1_stats['k'].sum(),
            f"Deaths {player1}": p1_stats['d'].sum(),
            f"Taxa K/D {player1}": (p1_stats['k'].sum() / p1_stats['d'].sum()) if p1_stats['d'].sum() > 0 else 0,
            f"Kills {player2}": p2_stats['k'].sum(),
            f"Deaths {player2}": p2_stats['d'].sum(),
            f"Taxa K/D {player2}": (p2_stats['k'].sum() / p2_stats['d'].sum()) if p2_stats['d'].sum() > 0 else 0
        }

    # --- FUNÇÃO H2H POR MAPA CORRIGIDA E MELHORADA ---
    def get_h2h_by_map(self, player1, player2, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        p1_vs_p2_matches = df[df['player'] == player1]['opponents'].apply(lambda x: player2 in x)
        match_ids = df[df['player'] == player1][p1_vs_p2_matches]['match_id'].unique()
        if len(match_ids) == 0: return pd.DataFrame()

        h2h_df = df[df['match_id'].isin(match_ids)]
        results = []
        for map_name, group in h2h_df.groupby('map'):
            p1_map_stats = group[group['player'] == player1]
            p2_map_stats = group[group['player'] == player2]
            
            if p1_map_stats.empty or p2_map_stats.empty: continue

            total_matches = group['match_id'].nunique()
            p1_wins = p1_map_stats.groupby('match_id')['won'].first().sum()
            p1_kd = (p1_map_stats['k'].sum() / p1_map_stats['d'].sum()) if p1_map_stats['d'].sum() > 0 else 0
            p2_kd = (p2_map_stats['k'].sum() / p2_map_stats['d'].sum()) if p2_map_stats['d'].sum() > 0 else 0

            results.append({
                'Mapa': map_name,
                'Partidas': total_matches,
                f'Vitórias {player1}': int(p1_wins),
                f'Vitórias {player2}": total_matches - int(p1_wins),
                f'K/D {player1}': p1_kd,
                f'K/D {player2}': p2_kd
            })
        return pd.DataFrame(results).sort_values(by='Partidas', ascending=False)

    # --- NOVA FUNÇÃO PARA ANÁLISE DE TIME 5X5 ---
    def get_team_stats(self, team_players, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        team_roster = frozenset(team_players)
        
        team_matches = df[df['team_roster'] == team_roster]
        if team_matches.empty:
            return None
        
        unique_matches = team_matches.groupby('match_id').first()
        total_games = len(unique_matches)
        wins = int(unique_matches['won'].sum())
        
        return {
            "Partidas Juntos": total_games,
            "Vitórias": wins,
            "Derrotas": total_games - wins,
            "% de Vitória": (wins / total_games) * 100 if total_games > 0 else 0
        }

# --- FUNÇÕES DE PLOTAGEM ---
def create_top_kd_chart(df):
    MIN_GAMES = 5
    filtered_df = df[df['Partidas Jogadas'] >= MIN_GAMES].nlargest(10, 'Taxa K/D')
    if filtered_df.empty: return None, f"Nenhum jogador com {MIN_GAMES} ou mais partidas para o gráfico."
    fig, ax = plt.subplots(figsize=(10, 6))
    players = filtered_df.index
    kd_ratio = filtered_df['Taxa K/D']
    ax.barh(players, kd_ratio, color='skyblue')
    for index, value in enumerate(kd_ratio):
        ax.text(value, index, f' {value:.2f}', va='center')
    ax.set_title(f'Top {len(filtered_df)} Jogadores por Taxa K/D (mín. {MIN_GAMES} partidas)', fontsize=14)
    ax.set_xlabel('Taxa K/D', fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return buf, None

def create_player_trend_chart(trend_df, player_name):
    if trend_df.empty: return None
    fig, ax1 = plt.subplots(figsize=(12, 6))
    match_index = range(1, len(trend_df) + 1)
    color = 'tab:blue'; ax1.set_xlabel('Partidas Jogadas (em ordem cronológica)'); ax1.set_ylabel('Taxa K/D Cumulativa', color=color); ax1.plot(match_index, trend_df['Taxa K/D Cumulativa'], color=color, marker='o', markersize=4); ax1.tick_params(axis='y', labelcolor=color); ax1.grid(True, axis='y', linestyle='--'); ax2 = ax1.twinx(); color = 'tab:red'; ax2.set_ylabel('% de Vitória Cumulativa', color=color); ax2.plot(match_index, trend_df['% de Vitória Cumulativa'], color=color, linestyle='--', marker='x', markersize=4); ax2.tick_params(axis='y', labelcolor=color); fig.suptitle(f'Tendência de Performance de {player_name}', fontsize=16); fig.tight_layout(rect=[0, 0.03, 1, 0.95]); buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(fig); return buf

# --- NOVA FUNÇÃO DE GRÁFICO PARA H2H ---
def create_h2h_comparison_chart(h2h_stats, player1, player2):
    metrics = ['Vitórias', 'Kills', 'Deaths', 'Taxa K/D']
    p1_values = [h2h_stats[f'Vitórias {player1}'], h2h_stats[f'Kills {player1}'], h2h_stats[f'Deaths {player1}'], h2h_stats[f'Taxa K/D {player1}']]
    p2_values = [h2h_stats[f'Vitórias {player2}'], h2h_stats[f'Kills {player2}'], h2h_stats[f'Deaths {player2}'], h2h_stats[f'Taxa K/D {player2}']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, p1_values, width, label=player1, color='cornflowerblue')
    rects2 = ax.bar(x + width/2, p2_values, width, label=player2, color='lightcoral')
    
    ax.set_ylabel('Valores')
    ax.set_title(f'Comparativo H2H: {player1} vs {player2}')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Adiciona rótulos nas barras
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}' if isinstance(height, float) and height < 5 else f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
    
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return buf

# --- INTERFACE GRÁFICA (Streamlit) ---
st.set_page_config(layout="wide", page_title="Análise de Partidas CS2")
st.title("📊 Painel de Análise de Partidas de Counter-Strike 2")

uploaded_file = st.file_uploader("Carregue seu arquivo 'match_data.txt'", type="txt")

if uploaded_file:
    try:
        string_data = uploaded_file.getvalue().decode("utf-8")
        match_data = [json.loads(line) for line in string_data.splitlines() if line.strip()]
        analyzer = StatsAnalyzer(match_data)
        player_list = analyzer.get_player_list()
        map_list = analyzer.get_map_list()

        st.sidebar.header("Filtros e Seleção")
        min_date, max_date = analyzer.df['date'].min().date(), analyzer.df['date'].max().date()
        date_range = st.sidebar.date_input("Selecione o Intervalo de Datas:", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        start_date, end_date = (date_range[0], date_range[1]) if len(date_range) == 2 else (min_date, max_date)
        
        st.sidebar.markdown("---")
        analysis_type = st.sidebar.radio("Tipo de Análise:", ("Estatísticas Gerais", "Análise por Mapa", "Análise de Jogador", "Análise de Dupla (Juntos)", "Confronto 1x1 (Contra)", "Análise de Time (5x5)"))

        if analysis_type == "Estatísticas Gerais":
            st.header(f"Estatísticas Gerais ({start_date} a {end_date})")
            st.subheader("Visualização: Top Jogadores por Taxa K/D")
            stats_df = analyzer.get_overall_player_stats(start_date, end_date)
            chart_buffer, error_msg = create_top_kd_chart(stats_df)
            if chart_buffer: st.image(chart_buffer)
            else: st.info(error_msg)
            
            st.subheader("Tabela Completa de Estatísticas")
            sort_option = st.selectbox("Ordenar por:", ['Partidas Jogadas', '% de Vitória', 'Taxa K/D', 'Saldo de Rounds', 'Abates (K)'])
            sorted_stats = analyzer.get_overall_player_stats(start_date, end_date, sort_by=sort_option)
            if not sorted_stats.empty: st.dataframe(sorted_stats.style.format({'% de Vitória': '{:.2f}%', 'Taxa K/D': '{:.2f}'}))
            else: st.info("Nenhuma partida encontrada no período.")

        elif analysis_type == "Confronto 1x1 (Contra)":
            st.sidebar.subheader("Selecione os Jogadores")
            player1 = st.sidebar.selectbox("Jogador 1:", player_list, key='h2h_p1')
            player2 = st.sidebar.selectbox("Jogador 2:", [p for p in player_list if p != player1], key='h2h_p2')
            
            if player1 and player2:
                st.header(f"Confronto Direto: {player1} vs {player2} ({start_date} a {end_date})")
                h2h_overall = analyzer.get_h2h_overall(player1, player2, start_date, end_date)
                if h2h_overall:
                    st.subheader("Resumo Geral do Confronto")
                    # GRÁFICO COMPARATIVO
                    h2h_chart = create_h2h_comparison_chart(h2h_overall, player1, player2)
                    st.image(h2h_chart)

                    st.subheader("Detalhes por Mapa")
                    h2h_map_stats = analyzer.get_h2h_by_map(player1, player2, start_date, end_date)
                    if not h2h_map_stats.empty: st.dataframe(h2h_map_stats.style.format({'K/D ' + player1: '{:.2f}', 'K/D ' + player2: '{:.2f}'}))
                    else: st.info("Nenhum confronto H2H encontrado nos mapas do período.")
                else:
                    st.warning("Estes jogadores nunca se enfrentaram no período selecionado.")

        elif analysis_type == "Análise de Time (5x5)":
            st.sidebar.subheader("Selecione 5 Jogadores")
            selected_players = st.sidebar.multiselect("Time:", player_list)
            
            if len(selected_players) == 5:
                team_name = " & ".join(sorted(selected_players))
                st.header(f"Análise do Time: {team_name} ({start_date} a {end_date})")
                
                team_stats = analyzer.get_team_stats(selected_players, start_date, end_date)
                if team_stats:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Partidas Juntos", team_stats["Partidas Juntos"])
                    c2.metric("Vitórias", team_stats["Vitórias"])
                    c3.metric("Derrotas", team_stats["Derrotas"])
                    c4.metric("% de Vitória", f"{team_stats['% de Vitória']:.2f}%")
                else:
                    st.warning("Este time de 5 jogadores nunca jogou junto no período selecionado.")
            elif len(selected_players) > 0:
                st.sidebar.warning("Por favor, selecione exatamente 5 jogadores.")

        # Outras abas (simplificado para não repetir tudo)
        elif analysis_type == "Análise de Jogador":
            player_name = st.sidebar.selectbox("Selecione o Jogador:", player_list)
            st.header(f"Análise Individual de {player_name}")
            # Coloque o resto da sua lógica de análise de jogador aqui...
            
    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo. Verifique o formato. Detalhe: {e}")
else:
    st.info("Aguardando o upload do arquivo `match_data.txt` para iniciar a análise.")
