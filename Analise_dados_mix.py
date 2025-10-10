import streamlit as st
import pandas as pd
import json
from collections import Counter
from datetime import datetime, date
import matplotlib.pyplot as plt
import io # Necessário para Streamlit não usar arquivos temporários diretamente

# --- LÓGICA DE ANÁLISE (CLASSE StatsAnalyzer) ---
# Adição de lógica para lidar com a filtragem temporal.

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
            
            # Time A
            for player_stats in match.get('team_a', []):
                player_name = self._unify_player_names(player_stats.get('player'))
                records.append({'match_id': match_id, 'date': match.get('date'), 'map': match.get('map'),'player': player_name, 'k': player_stats.get('k', 0), 'a': player_stats.get('a', 0), 'd': player_stats.get('d', 0),'team': 'A', 'won': is_team_a_winner, 'round_diff': score_a - score_b,'rounds_ganhos': score_a, 'rounds_perdidos': score_b,'teammates': [self._unify_player_names(p.get('player')) for p in match.get('team_a', []) if p.get('player') != player_stats.get('player')],'opponents': [self._unify_player_names(p.get('player')) for p in match.get('team_b', [])]})
            
            # Time B
            for player_stats in match.get('team_b', []):
                player_name = self._unify_player_names(player_stats.get('player'))
                records.append({'match_id': match_id, 'date': match.get('date'), 'map': match.get('map'),'player': player_name, 'k': player_stats.get('k', 0), 'a': player_stats.get('a', 0), 'd': player_stats.get('d', 0),'team': 'B', 'won': is_team_a_winner is not None and not is_team_a_winner, 'round_diff': score_b - score_a,'rounds_ganhos': score_b, 'rounds_perdidos': score_a,'teammates': [self._unify_player_names(p.get('player')) for p in match.get('team_b', []) if p.get('player') != player_stats.get('player')],'opponents': [self._unify_player_names(p.get('player')) for p in match.get('team_a', [])]})
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def _filter_by_date(self, start_date, end_date):
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())
        filtered_df = self.df[(self.df['date'] >= start_dt) & (self.df['date'] <= end_dt)]
        return filtered_df
    
    # --- Métodos de Listagem ---
    def get_player_list(self): return sorted(self.df['player'].unique())
    def get_map_list(self): return sorted(self.df['map'].unique())
    
    # --- NOVO MÉTODO PARA TENDÊNCIA TEMPORAL ---
    def get_player_cumulative_trend(self, player_name, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        player_df = df[df['player'] == player_name].sort_values(by='date').copy()
        
        if player_df.empty:
            return pd.DataFrame()
        
        # Agrega estatísticas por partida (match_id) para evitar duplicação por data
        match_stats = player_df.groupby('match_id').agg({
            'date': 'first',
            'k': 'first', 
            'd': 'first',
            'won': 'first'
        }).sort_values(by='date').reset_index()

        # Cálculo Cumulativo de K/D
        match_stats['K_Cumulativo'] = match_stats['k'].cumsum()
        match_stats['D_Cumulativo'] = match_stats['d'].cumsum()
        match_stats['Taxa K/D Cumulativa'] = (match_stats['K_Cumulativo'] / match_stats['D_Cumulativo']).fillna(0)
        
        # Cálculo Cumulativo de % de Vitória
        match_stats['Vitórias_Cumulativas'] = match_stats['won'].cumsum()
        match_stats['Partidas_Totais'] = range(1, len(match_stats) + 1)
        match_stats['% de Vitória Cumulativa'] = (match_stats['Vitórias_Cumulativas'] / match_stats['Partidas_Totais']) * 100
        
        return match_stats[['date', 'match_id', 'Taxa K/D Cumulativa', '% de Vitória Cumulativa']]


    # --- MÉTODOS DE ANÁLISE (EXISTENTES, MODIFICADOS PARA USAR FILTRO DE DATA) ---

    def get_overall_player_stats(self, start_date, end_date, sort_by='Partidas Jogadas'):
        df = self._filter_by_date(start_date, end_date)
        if df.empty: return pd.DataFrame()
        
        stats = df.groupby('player').agg(**{'Partidas Jogadas': ('match_id', 'nunique'),'Abates (K)': ('k', 'sum'),'Assistências (A)': ('a', 'sum'),'Mortes (D)': ('d', 'sum'), 'Saldo de Rounds': ('round_diff', 'sum')})
        wins = df[df['won'] == True].groupby('player')['won'].count()
        losses = df[df['won'] == False].groupby('player')['won'].count()
        stats['Vitórias'] = wins.reindex(stats.index, fill_value=0)
        stats['Derrotas'] = losses.reindex(stats.index, fill_value=0)
        stats['% de Vitória'] = (stats['Vitórias'] / (stats['Vitórias'] + stats['Derrotas'])).fillna(0) * 100
        stats['Taxa K/D'] = (stats['Abates (K)'] / stats['Mortes (D)']).fillna(0)
        stats = stats[['Partidas Jogadas', 'Vitórias', 'Derrotas', '% de Vitória', 'Abates (K)', 'Assistências (A)', 'Mortes (D)', 'Taxa K/D', 'Saldo de Rounds']]
        return stats.sort_values(by=sort_by, ascending=False)

    def get_map_leaderboard(self, map_name, start_date, end_date, sort_by='% de Vitória'):
        df = self._filter_by_date(start_date, end_date)
        map_df = df[df['map'] == map_name]
        if map_df.empty: return pd.DataFrame()
        
        stats = map_df.groupby('player').agg(**{'Partidas Jogadas': ('match_id', 'nunique'), 'Abates (K)': ('k', 'sum'), 'Mortes (D)': ('d', 'sum'), 'Saldo de Rounds': ('round_diff', 'sum')})
        wins = map_df[map_df['won'] == True].groupby('player')['won'].count()
        losses = map_df[map_df['won'] == False].groupby('player')['won'].count()
        stats['Vitórias'] = wins.reindex(stats.index, fill_value=0)
        stats['Derrotas'] = losses.reindex(stats.index, fill_value=0)
        stats['% de Vitória'] = (stats['Vitórias'] / (stats['Vitórias'] + stats['Derrotas'])).fillna(0) * 100
        stats['Taxa K/D'] = (stats['Abates (K)'] / stats['Mortes (D)']).fillna(0)
        return stats.sort_values(by=sort_by, ascending=False)
        
    def get_player_overall_stats_summary(self, player_name, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        player_df = df[df['player'] == player_name].dropna(subset=['won'])
        if player_df.empty: return None
        total_games = len(player_df.groupby('match_id').first())
        wins = player_df.groupby('match_id').first()['won'].sum()
        total_rounds_ganhos = player_df['rounds_ganhos'].sum()
        total_rounds_perdidos = player_df['rounds_perdidos'].sum()
        round_diff = player_df['round_diff'].sum()
        
        win_rate = (wins / total_games) * 100 if total_games > 0 else 0
        
        return {"Partidas Jogadas": total_games, "Vitórias": int(wins), "Derrotas": total_games - int(wins), "% de Vitória": f"{win_rate:.2f}%", "Rounds Ganhos": total_rounds_ganhos, "Rounds Perdidos": total_rounds_perdidos, "Saldo de Rounds": round_diff}

    def get_performance_by_map(self, player_name, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        player_df = df[df['player'] == player_name]
        if player_df.empty: return pd.DataFrame()
        
        stats_by_map = player_df.groupby('map').agg(
            Partidas=('match_id', 'nunique'), 
            Abates=('k', 'sum'), 
            Mortes=('d', 'sum'),
            Rounds_Ganhos=('rounds_ganhos', 'sum'),
            Rounds_Perdidos=('rounds_perdidos', 'sum'),
            Saldo_Rounds=('round_diff', 'sum')
        )
        
        wins_by_map = player_df[player_df['won'] == True].groupby('map')['won'].count()
        
        stats_by_map['Vitórias'] = wins_by_map.reindex(stats_by_map.index, fill_value=0)
        stats_by_map['% de Vitória'] = (stats_by_map['Vitórias'] / stats_by_map['Partidas']) * 100
        stats_by_map['K/D'] = (stats_by_map['Abates'] / stats_by_map['Mortes']).fillna(0)
        
        stats_by_map = stats_by_map[['Partidas', 'Vitórias', '% de Vitória', 'Abates', 'Mortes', 'K/D', 'Rounds_Ganhos', 'Rounds_Perdidos', 'Saldo_Rounds']]
        
        return stats_by_map.sort_values(by='Partidas', ascending=False)
    
    # MÉTODOS DE DUPLA E H2H (mantidos com filtro de data)
    def get_performance_with_teammates(self, player_name, start_date, end_date, best=True):
        df = self._filter_by_date(start_date, end_date)
        player_matches = df[df['player'] == player_name]
        if player_matches.empty: return pd.DataFrame()
        teammate_list = player_matches['teammates'].explode()
        common_teammates = teammate_list.value_counts()
        if common_teammates.empty: return pd.DataFrame()
        performance = []
        for teammate, games_played in common_teammates.items():
            matches_with_teammate = df[(df['player'] == player_name) & (df['teammates'].apply(lambda x: teammate in x))].dropna(subset=['won'])
            unique_matches = matches_with_teammate.groupby('match_id').first()
            if not unique_matches.empty:
                wins = unique_matches['won'].sum()
                total_matches = len(unique_matches)
                win_rate = (wins / total_matches) * 100
                performance.append({'Companheiro': teammate, 'Partidas Juntos': total_matches, '% de Vitória Juntos': win_rate})
        if not performance: return pd.DataFrame()
        return pd.DataFrame(performance).sort_values(by='% de Vitória Juntos', ascending=not best)

    def get_duo_stats(self, player1, player2, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        duo_matches = df[(df['player'] == player1) & (df['teammates'].apply(lambda x: player2 in x))].dropna(subset=['won'])
        if duo_matches.empty: return None
        total_games = len(duo_matches.groupby('match_id').first())
        wins = duo_matches.groupby('match_id').first()['won'].sum()
        win_rate = (wins / total_games) * 100
        p1_stats = duo_matches[['k', 'a', 'd']].sum()
        p2_matches = df[(df['player'] == player2) & (df['match_id'].isin(duo_matches['match_id']))]
        p2_stats = p2_matches[['k', 'a', 'd']].sum()
        combined_stats = p1_stats + p2_stats
        combined_kd = (combined_stats['k'] / combined_stats['d']) if combined_stats['d'] > 0 else 0
        return {"Partidas Juntos": total_games, "Vitórias": int(wins), "Derrotas": total_games - int(wins), "% de Vitória da Dupla": f"{win_rate:.2f}%", "K/D Combinado": f"{combined_kd:.2f}"}

    def get_h2h_overall(self, player1, player2, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        h2h_matches_p1 = df[(df['player'] == player1) & (df['opponents'].apply(lambda x: player2 in x))].dropna(subset=['won'])
        if h2h_matches_p1.empty: return None
        unique_matches = h2h_matches_p1.groupby('match_id').first()
        p1_wins = unique_matches['won'].sum(); total_games = len(unique_matches); p2_wins = total_games - p1_wins
        p1_total_k = h2h_matches_p1['k'].sum(); p1_total_d = h2h_matches_p1['d'].sum(); p1_kd = p1_total_k / p1_total_d if p1_total_d > 0 else 0
        h2h_matches_p2 = df[(df['player'] == player2) & (df['match_id'].isin(h2h_matches_p1['match_id']))]
        p2_total_k = h2h_matches_p2['k'].sum(); p2_total_d = h2h_matches_p2['d'].sum(); p2_kd = p2_total_k / p2_total_d if p2_total_d > 0 else 0
        return {"Partidas H2H": total_games, f"Vitórias {player1}": int(p1_wins), f"Vitórias {player2}": int(p2_wins), f"K/D Total {player1}": f"{p1_total_k}/{p1_total_d}", f"Taxa K/D {player1}": f"{p1_kd:.2f}", f"K/D Total {player2}": f"{p2_total_k}/{p2_total_d}", f"Taxa K/D {player2}": f"{p2_kd:.2f}"}

def get_h2h_by_map(self, player1, player2, start_date, end_date):
    df = self._filter_by_date(start_date, end_date)
    h2h_matches_p1 = df[(df['player'] == player1) & (df['opponents'].apply(lambda x: player2 in x))]
    if h2h_matches_p1.empty: return pd.DataFrame()
    results = []
    for map_name, group in h2h_matches_p1.groupby('map'):
        match_stats = group.groupby('match_id').first()
        p1_stats = group.agg(K=('k', 'sum'), D=('d', 'sum'), RoundDiff=('round_diff', 'sum'))
        p1_wins = match_stats['won'].sum()
        total_matches = len(match_stats)
        p2_matches_on_map = df[(df['player'] == player2) & (df['match_id'].isin(group['match_id']))]
        p2_stats = p2_matches_on_map.agg(K=('k', 'sum'), D=('d', 'sum'))
        
        # LINHA CORRIGIDA ABAIXO
        results.append({'Mapa': map_name, 'Partidas': total_matches, 
                        f'Vitórias {player1}': int(p1_wins), 
                        f'Vitórias {player2}': total_matches - int(p1_wins), 
                        f'K/D {player1}': f"{p1_stats['K']}/{p1_stats['D']}", 
                        f'K/D {player2}': f"{p2_stats['K']}/{p2_stats['D']}", 
                        'Saldo de Rounds (p/ P1)': p1_stats['RoundDiff']})
    return pd.DataFrame(results)

# --- FUNÇÕES DE PLOTAGEM ---
def create_top_kd_chart(df):
    """Cria um gráfico de barras para o Top 10 K/D, filtrando por partidas mínimas."""
    MIN_GAMES = 5
    filtered_df = df[df['Partidas Jogadas'] >= MIN_GAMES].nlargest(10, 'Taxa K/D')
    
    if filtered_df.empty:
        return None, f"Nenhum jogador com mais de {MIN_GAMES} partidas para o gráfico."
        
    fig, ax = plt.subplots(figsize=(10, 6))
    players = filtered_df.index
    kd_ratio = filtered_df['Taxa K/D']
    
    ax.barh(players, kd_ratio, color='skyblue')
    
    # Adiciona a Taxa K/D como rótulo na barra
    for index, value in enumerate(kd_ratio):
        ax.text(value, index, f'{value:.2f}', va='center')
        
    ax.set_title(f'Top {len(filtered_df)} Jogadores por Taxa K/D (mínimo de {MIN_GAMES} partidas)', fontsize=14)
    ax.set_xlabel('Taxa K/D', fontsize=12)
    ax.set_ylabel('Jogador', fontsize=12)
    plt.gca().invert_yaxis() # Coloca o Top 1 no topo
    plt.tight_layout()
    
    # Salva o gráfico em um buffer (para Streamlit)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return buf, None

def create_player_trend_chart(trend_df, player_name):
    """Cria um gráfico de linha para a tendência cumulativa de K/D e % Vitória."""
    if trend_df.empty:
        return None
        
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    match_index = trend_df.index # Usaremos o índice (partidas em ordem)
    
    # Eixo Y1: Taxa K/D Cumulativa
    color = 'tab:blue'
    ax1.set_xlabel('Partidas Jogadas')
    ax1.set_ylabel('Taxa K/D Cumulativa', color=color)
    ax1.plot(match_index, trend_df['Taxa K/D Cumulativa'], color=color, label='K/D')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, axis='y', linestyle='--')
    
    # Eixo Y2: % de Vitória Cumulativa
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('% de Vitória Cumulativa', color=color)  
    ax2.plot(match_index, trend_df['% de Vitória Cumulativa'], color=color, linestyle='--', label='% Vitória')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.suptitle(f'Tendência de Performance Cumulativa de {player_name}', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajusta para o suptitle
    
    # Salva o gráfico em um buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return buf

# --- INTERFACE GRÁFICA (Streamlit) ---

st.set_page_config(layout="wide", page_title="Análise de Partidas CS2 - Gráficos")
st.title("📊 Painel de Análise de Partidas de Counter-Strike 2")

uploaded_file = st.file_uploader("Carregue seu arquivo 'match_data.txt'", type="txt")

if uploaded_file is not None:
    try:
        string_data = uploaded_file.getvalue().decode("utf-8")
        match_data = [json.loads(line) for line in string_data.splitlines() if line.strip()]
        
        analyzer = StatsAnalyzer(match_data)
        player_list = analyzer.get_player_list()
        map_list = analyzer.get_map_list()

        # --- SELEÇÃO DE DATA E TIPO DE ANÁLISE NA SIDEBAR ---
        st.sidebar.header("Filtros e Seleção")
        
        min_date = analyzer.df['date'].min().date()
        max_date = analyzer.df['date'].max().date()

        date_range = st.sidebar.date_input(
            "Selecione o Intervalo de Datas:", 
            value=(min_date, max_date), 
            min_value=min_date, 
            max_value=max_date
        )
        
        start_date = date_range[0] if len(date_range) == 2 else min_date
        end_date = date_range[1] if len(date_range) == 2 else max_date
            
        st.sidebar.markdown("---")
        analysis_type = st.sidebar.radio("Tipo de Análise:", ("Estatísticas Gerais", "Análise por Mapa", "Análise de Jogador", "Análise de Dupla (Juntos)", "Confronto 1x1 (Contra)"))

        
        # --- LÓGICA DE EXIBIÇÃO DE DADOS (COM GRÁFICOS) ---
        
        if analysis_type == "Estatísticas Gerais":
            st.header(f"Estatísticas Gerais de Todos os Jogadores ({start_date} a {end_date})")
            
            # --- NOVO GRÁFICO 1: TOP K/D ---
            st.subheader("Visualização: Top Jogadores por Taxa K/D")
            stats_df = analyzer.get_overall_player_stats(start_date, end_date, sort_by='Taxa K/D')
            
            chart_buffer, error_msg = create_top_kd_chart(stats_df)
            
            if chart_buffer:
                # Usa st.image para exibir o gráfico do buffer
                st.image(chart_buffer, caption='Taxa K/D para jogadores com no mínimo 5 partidas.')
            else:
                st.info(error_msg)
            
            # --- TABELA GERAL ---
            st.subheader("Tabela Completa de Estatísticas")
            sort_option = st.selectbox("Ordenar por:", ['Partidas Jogadas', '% de Vitória', 'Taxa K/D', 'Saldo de Rounds', 'Abates (K)'])
            stats_df = analyzer.get_overall_player_stats(start_date, end_date, sort_by=sort_option)
            if not stats_df.empty:
                st.dataframe(stats_df.style.format({'% de Vitória': '{:.2f}%', 'Taxa K/D': '{:.2f}'}))
            else:
                st.info("Nenhuma partida encontrada no intervalo de datas selecionado.")
        
        elif analysis_type == "Análise por Mapa":
            st.header(f"Ranking de Jogadores por Mapa ({start_date} a {end_date})")
            selected_map = st.selectbox("Selecione um Mapa:", map_list)
            if selected_map:
                st.subheader(f"Melhores jogadores em: {selected_map}")
                sort_option_map = st.selectbox("Ordenar por:", ['Partidas Jogadas', '% de Vitória', 'Taxa K/D', 'Saldo de Rounds'], key='map_sort')
                map_leaderboard = analyzer.get_map_leaderboard(selected_map, start_date, end_date, sort_by=sort_option_map)
                if not map_leaderboard.empty:
                    st.dataframe(map_leaderboard.style.format({'% de Vitória': '{:.2f}%', 'Taxa K/D': '{:.2f}'}))
                else:
                    st.info("Nenhuma partida neste mapa encontrada no intervalo de datas selecionado.")

        elif analysis_type == "Análise de Jogador":
            player_name = st.sidebar.selectbox("Selecione o Jogador:", player_list)
            if player_name:
                st.header(f"Análise Individual de {player_name} ({start_date} a {end_date})")
                overall_stats = analyzer.get_player_overall_stats_summary(player_name, start_date, end_date)
                if overall_stats:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Partidas", overall_stats["Partidas Jogadas"])
                    c2.metric("Vitórias", overall_stats["Vitórias"])
                    c3.metric("Derrotas", overall_stats["Derrotas"])
                    c4.metric("% de Vitória", overall_stats["% de Vitória"])
                    
                    st.subheader("Performance de Rounds Ganhos/Perdidos")
                    c5, c6, c7 = st.columns(3)
                    c5.metric("Rounds Ganhos", overall_stats["Rounds Ganhos"])
                    c6.metric("Rounds Perdidos", overall_stats["Rounds Perdidos"])
                    c7.metric("Saldo de Rounds", overall_stats["Saldo de Rounds"])

                    # --- NOVO GRÁFICO 2: TENDÊNCIA CUMULATIVA ---
                    st.subheader("Tendência de Performance ao Longo do Tempo")
                    trend_df = analyzer.get_player_cumulative_trend(player_name, start_date, end_date)
                    trend_chart_buffer = create_player_trend_chart(trend_df, player_name)
                    if trend_chart_buffer:
                        st.image(trend_chart_buffer, caption='Evolução da Taxa K/D e % de Vitória cumulativa por partida.')
                    else:
                        st.info("Não há dados de partidas suficientes para gerar o gráfico de tendência neste intervalo.")

                    st.subheader("Desempenho Detalhado por Mapa")
                    map_perf_df = analyzer.get_performance_by_map(player_name, start_date, end_date)
                    st.dataframe(map_perf_df.style.format({'% de Vitória': '{:.2f}%', 'K/D': '{:.2f}'}))
                    
                    st.subheader("Análise de Companheiros")
                    c_best, c_worst = st.columns(2)
                    with c_best: st.subheader("Melhores Companheiros (% de Vitória)"); st.dataframe(analyzer.get_performance_with_teammates(player_name, start_date, end_date, best=True).head(5))
                    with c_worst: st.subheader("Piores Companheiros (% de Vitória)"); st.dataframe(analyzer.get_performance_with_teammates(player_name, start_date, end_date, best=False).head(5))
                else:
                    st.info("Nenhuma partida encontrada para este jogador no intervalo de datas selecionado.")

        elif analysis_type == "Análise de Dupla (Juntos)":
            st.sidebar.subheader("Selecione a Dupla")
            player1 = st.sidebar.selectbox("Jogador 1:", player_list, index=player_list.index("Khan") if "Khan" in player_list else 0)
            player2 = st.sidebar.selectbox("Jogador 2:", player_list, index=player_list.index("Fzr") if "Fzr" in player_list and "Fzr" != player1 else (player_list.index(player_list[1]) if len(player_list)>1 and player_list[1] != player1 else 0))
            if player1 and player2 and player1 != player2:
                st.header(f"Análise da Dupla: {player1} & {player2} ({start_date} a {end_date})"); duo_stats = analyzer.get_duo_stats(player1, player2, start_date, end_date)
                if duo_stats:
                    c1, c2, c3, c4, c5 = st.columns(5); c1.metric("Partidas Juntos", duo_stats["Partidas Juntos"]); c2.metric("Vitórias", duo_stats["Vitórias"]); c3.metric("Derrotas", duo_stats["Derrotas"]); c4.metric("% de Vitória", duo_stats["% de Vitória da Dupla"]); c5.metric("K/D Combinado", duo_stats["K/D Combinado"])
                else: st.warning("Esta dupla nunca jogou junta no intervalo de datas selecionado.")
        
        elif analysis_type == "Confronto 1x1 (Contra)":
            st.sidebar.subheader("Selecione os Jogadores")
            player1 = st.sidebar.selectbox("Jogador 1:", player_list, index=player_list.index("Khan") if "Khan" in player_list else 0, key='h2h_p1')
            player2 = st.sidebar.selectbox("Jogador 2:", player_list, index=player_list.index("Ace") if "Ace" in player_list and "Ace" != player1 else (player_list.index(player_list[1]) if len(player_list)>1 and player_list[1] != player1 else 0), key='h2h_p2')
            if player1 and player2 and player1 != player2:
                st.header(f"Confronto Direto: {player1} vs {player2} ({start_date} a {end_date})")
                h2h_overall = analyzer.get_h2h_overall(player1, player2, start_date, end_date)
                if h2h_overall:
                    st.subheader("Resumo Geral do Confronto")
                    c1, c2, c3 = st.columns(3); c1.metric("Partidas H2H", h2h_overall["Partidas H2H"]); c2.metric(f"Vitórias {player1}", h2h_overall[f"Vitórias {player1}"]); c3.metric(f"Vitórias {player2}", h2h_overall[f"Vitórias {player2}"])
                    c1, c2 = st.columns(2); c1.metric(f"Taxa K/D {player1}", h2h_overall[f"Taxa K/D {player1}"], help=h2h_overall[f"K/D Total {player1}"]); c2.metric(f"Taxa K/D {player2}", h2h_overall[f"Taxa K/D {player2}"], help=h2h_overall[f"K/D Total {player2}"])
                    st.subheader("Detalhes por Mapa"); h2h_map_stats = analyzer.get_h2h_by_map(player1, player2, start_date, end_date)
                    if not h2h_map_stats.empty:
                        st.dataframe(h2h_map_stats)
                    else:
                        st.info("Nenhum confronto neste intervalo de datas foi encontrado nos mapas.")
                else: st.warning("Estes jogadores nunca se enfrentaram no intervalo de datas selecionado.")

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo. Verifique se o formato está correto. Detalhe: {e}")

else:
    st.info("Aguardando o upload do arquivo `match_data.txt` para iniciar a análise.")



