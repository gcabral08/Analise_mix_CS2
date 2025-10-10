import streamlit as st
import pandas as pd
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import io
from sklearn.linear_model import LinearRegression

# --- LÓGICA DE ANÁLISE (CLASSE StatsAnalyzer) ---
class StatsAnalyzer:
    def __init__(self, data):
        if not data:
            raise ValueError("Os dados das partidas não foram carregados.")
        self.raw_data = data
        self.df = self._load_data_into_dataframe()
        self._precompute_rosters()

    def _unify_player_names(self, name):
        name_map = {'Birigul': 'Birigu1', 'Craque Perfil Líder': 'Craque Perfil Lider', 'F1dellis': 'Fidellis', 'BOSSS_': 'BOSSS', 'FzrG': 'Fzr'}
        return name_map.get(name, name)

    def _load_data_into_dataframe(self):
        records = []
        for match_id, match in enumerate(self.raw_data):
            score_a, score_b = match.get('score_a', 0), match.get('score_b', 0)
            is_team_a_winner = True if score_a > score_b else (False if score_b > score_a else None)
            
            team_a_players = {self._unify_player_names(p.get('player')) for p in match.get('team_a', [])}
            team_b_players = {self._unify_player_names(p.get('player')) for p in match.get('team_b', [])}

            for team_players, winner, score_diff, r_ganhos, r_perdidos, opponents in [
                (team_a_players, is_team_a_winner, score_a - score_b, score_a, score_b, team_b_players),
                (team_b_players, is_team_a_winner is not None and not is_team_a_winner, score_b - score_a, score_b, score_a, team_a_players)
            ]:
                original_team_data = match.get('team_a') if team_players == team_a_players else match.get('team_b')
                for player_stats in original_team_data:
                    player_name = self._unify_player_names(player_stats.get('player'))
                    records.append({
                        'match_id': match_id, 'date': match.get('date'), 'map': match.get('map'), 'player': player_name,
                        'k': player_stats.get('k', 0), 'a': player_stats.get('a', 0), 'd': player_stats.get('d', 0),
                        'team_roster': frozenset(team_players), 'won': winner, 'round_diff': score_diff,
                        'rounds_ganhos': r_ganhos, 'rounds_perdidos': r_perdidos, 'opponents': opponents
                    })
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def _precompute_rosters(self):
        if self.df.empty:
            self.rosters_by_match = pd.DataFrame()
            return
        self.rosters_by_match = self.df.groupby('match_id').agg(
            team_roster=('team_roster', 'first'),
            opponents=('opponents', 'first'),
            date=('date', 'first'),
            map=('map', 'first'),
            rounds_ganhos=('rounds_ganhos', 'first'),
            rounds_perdidos=('rounds_perdidos', 'first'),
            won=('won', 'first'),
            round_diff=('round_diff', 'first')
        )

    def _filter_by_date(self, start_date, end_date):
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())
        return self.df[(self.df['date'] >= start_dt) & (self.df['date'] <= end_dt)]

    def get_player_list(self): return sorted(self.df['player'].unique())
    def get_map_list(self): return sorted(self.df['map'].unique())

    def get_overall_player_stats(self, start_date, end_date, sort_by='Partidas Jogadas'):
        df = self._filter_by_date(start_date, end_date)
        if df.empty: return pd.DataFrame()
        stats = df.groupby('player').agg(
            **{'Partidas Jogadas': ('match_id', 'nunique'), 'Abates (K)': ('k', 'sum'),
               'Mortes (D)': ('d', 'sum'), 'Saldo de Rounds': ('round_diff', 'sum')})
        wins = df[df['won'] == True].groupby('player')['match_id'].nunique()
        stats['Vitórias'] = wins.reindex(stats.index, fill_value=0).astype(int)
        stats['Derrotas'] = stats['Partidas Jogadas'] - stats['Vitórias']
        stats['% de Vitória'] = (stats['Vitórias'] / stats['Partidas Jogadas']).fillna(0) * 100
        stats['Taxa K/D'] = (stats['Abates (K)'] / stats['Mortes (D)']).replace([np.inf, -np.inf], 0).fillna(0)
        return stats.sort_values(by=sort_by, ascending=False)
        
    def get_player_overall_stats_summary(self, player_name, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        player_df = df[df['player'] == player_name]
        if player_df.empty: return None
        match_stats = player_df.groupby('match_id').first()
        total_games = len(match_stats); wins = int(match_stats['won'].sum())
        return {
            "Partidas Jogadas": total_games, "Vitórias": wins, "Derrotas": total_games - wins,
            "% de Vitória": f"{(wins / total_games * 100):.2f}%" if total_games > 0 else "0.00%",
            "Rounds Ganhos": int(match_stats['rounds_ganhos'].sum()),
            "Rounds Perdidos": int(match_stats['rounds_perdidos'].sum()),
            "Saldo de Rounds": int(match_stats['round_diff'].sum())
        }

    def get_player_cumulative_trend(self, player_name, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        player_df = df[df['player'] == player_name].sort_values(by='date')
        if player_df.empty: return pd.DataFrame()
        match_stats = player_df.groupby('match_id').agg(date=('date', 'first'), k=('k', 'first'), d=('d', 'first'), won=('won', 'first')).sort_values(by='date')
        match_stats['K_Cumulativo'] = match_stats['k'].cumsum()
        match_stats['D_Cumulativo'] = match_stats['d'].cumsum()
        match_stats['Taxa K/D Cumulativa'] = (match_stats['K_Cumulativo'] / match_stats['D_Cumulativo']).fillna(0)
        match_stats['Vitórias_Cumulativas'] = match_stats['won'].cumsum()
        match_stats['Partidas_Totais'] = range(1, len(match_stats) + 1)
        match_stats['% de Vitória Cumulativa'] = (match_stats['Vitórias_Cumulativas'] / match_stats['Partidas_Totais']) * 100
        return match_stats

    def get_performance_with_teammates(self, player_name, start_date, end_date, best=True):
        df = self._filter_by_date(start_date, end_date)
        player_matches = df[df['player'] == player_name]
        if player_matches.empty: return pd.DataFrame()
        teammate_records = [{'match_id': mid, 'teammate': t, 'won': row['won']} for mid, row in player_matches.groupby('match_id').first().iterrows() for t in row['team_roster'] if t != player_name]
        if not teammate_records: return pd.DataFrame()
        teammate_df = pd.DataFrame(teammate_records)
        stats = teammate_df.groupby('teammate').agg(Partidas_Juntos=('match_id', 'nunique'), Vitórias=('won', 'sum'))
        stats['% de Vitória Juntos'] = (stats['Vitórias'] / stats['Partidas_Juntos']) * 100
        return stats.sort_values(by='% de Vitória Juntos', ascending=not best).head(5)

    def get_map_leaderboard(self, map_name, start_date, end_date, sort_by='% de Vitória'):
        df = self._filter_by_date(start_date, end_date)
        map_df = df[df['map'] == map_name]
        if map_df.empty: return pd.DataFrame()
        stats = map_df.groupby('player').agg(
            **{'Partidas Jogadas': ('match_id', 'nunique'), 'Abates (K)': ('k', 'sum'),
               'Mortes (D)': ('d', 'sum'), 'Saldo de Rounds': ('round_diff', 'sum')})
        wins = map_df[map_df['won'] == True].groupby('player')['match_id'].nunique()
        stats['Vitórias'] = wins.reindex(stats.index, fill_value=0).astype(int)
        stats['Derrotas'] = stats['Partidas Jogadas'] - stats['Vitórias']
        stats['% de Vitória'] = (stats['Vitórias'] / stats['Partidas Jogadas']).fillna(0) * 100
        stats['Taxa K/D'] = (stats['Abates (K)'] / stats['Mortes (D)']).replace([np.inf, -np.inf], 0).fillna(0)
        return stats.sort_values(by=sort_by, ascending=False)

    def get_h2h_overall(self, player1, player2, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        p1_rows = df[df['player'] == player1]
        match_ids = p1_rows[p1_rows['opponents'].apply(lambda o: player2 in o)]['match_id'].unique()
        if len(match_ids) == 0: return None
        h2h_df = df[df['match_id'].isin(match_ids)]
        p1_stats, p2_stats = h2h_df[h2h_df['player'] == player1], h2h_df[h2h_df['player'] == player2]
        p1_wins = p1_stats.groupby('match_id')['won'].first().sum()
        return { "Partidas H2H": len(match_ids), f"Vitórias {player1}": int(p1_wins), f"Vitórias {player2}": len(match_ids) - int(p1_wins), f"Kills {player1}": p1_stats['k'].sum(), f"Deaths {player1}": p1_stats['d'].sum(), f"Taxa K/D {player1}": (p1_stats['k'].sum() / p1_stats['d'].sum()) if p1_stats['d'].sum() > 0 else 0, f"Kills {player2}": p2_stats['k'].sum(), f"Deaths {player2}": p2_stats['d'].sum(), f"Taxa K/D {player2}": (p2_stats['k'].sum() / p2_stats['d'].sum()) if p2_stats['d'].sum() > 0 else 0 }

    def get_h2h_by_map(self, player1, player2, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        p1_rows = df[df['player'] == player1]
        match_ids = p1_rows[p1_rows['opponents'].apply(lambda o: player2 in o)]['match_id'].unique()
        if len(match_ids) == 0: return pd.DataFrame()
        h2h_df = df[df['match_id'].isin(match_ids)]
        results = []
        for map_name, group in h2h_df.groupby('map'):
            p1_map_stats, p2_map_stats = group[group['player'] == player1], group[group['player'] == player2]
            if p1_map_stats.empty or p2_map_stats.empty: continue
            total_matches, p1_wins = group['match_id'].nunique(), p1_map_stats.groupby('match_id')['won'].first().sum()
            p1_kd = (p1_map_stats['k'].sum() / p1_map_stats['d'].sum()) if p1_map_stats['d'].sum() > 0 else 0
            p2_kd = (p2_map_stats['k'].sum() / p2_map_stats['d'].sum()) if p2_map_stats['d'].sum() > 0 else 0
            results.append({ 'Mapa': map_name, 'Partidas': total_matches, f'Vitórias {player1}': int(p1_wins), f'Vitórias {player2}': total_matches - int(p1_wins), f'K/D {player1}': p1_kd, f'K/D {player2}': p2_kd })
        return pd.DataFrame(results).sort_values(by='Partidas', ascending=False)

    def get_core_player_stats(self, core_players, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        all_players = sorted(df['player'].unique())
        if not core_players:
            return {"Partidas Juntos": 0, "% de Vitória": 0}, all_players, pd.DataFrame()
        core_set = set(core_players)
        relevant_rosters = self.rosters_by_match[self.rosters_by_match.index.isin(df['match_id'].unique())]
        valid_match_ids = [mid for mid, row in relevant_rosters.iterrows() if core_set.issubset(row['team_roster'])]
        if not valid_match_ids:
            return {"Partidas Juntos": 0, "% de Vitória": 0}, [], pd.DataFrame()
        match_info = self.rosters_by_match.loc[valid_match_ids]
        total_games = len(match_info)
        wins = int(match_info['won'].sum())
        stats = {"Partidas Juntos": total_games, "Vitórias": wins, "Derrotas": total_games - wins, "% de Vitória": (wins / total_games * 100) if total_games > 0 else 0}
        possible_next_players = set()
        for roster in match_info['team_roster']:
            possible_next_players.update(roster)
        history = match_info.copy()
        history['Oponentes'] = history['opponents'].apply(lambda x: ', '.join(sorted(list(x))))
        history['Placar'] = history.apply(lambda row: f"{int(row['rounds_ganhos'])} a {int(row['rounds_perdidos'])}" if row['won'] else f"{int(row['rounds_perdidos'])} a {int(row['rounds_ganhos'])}", axis=1)
        history.rename(columns={'map': 'Mapa', 'date': 'Data'}, inplace=True)
        history['Data'] = history['Data'].dt.strftime('%Y-%m-%d')
        return stats, sorted(list(possible_next_players - core_set)), history[['Data', 'Mapa', 'Placar', 'Oponentes']]

    def get_match_count_over_time(self, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        if df.empty: return pd.Series(dtype='int64')
        unique_matches = df.drop_duplicates(subset=['match_id'])
        return unique_matches.groupby(unique_matches['date'].dt.date)['match_id'].count()
        
    def get_team_combination_stats(self, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        if df.empty: return pd.DataFrame()
        unique_matches_by_team = self.rosters_by_match[self.rosters_by_match.index.isin(df['match_id'].unique())]
        if unique_matches_by_team.empty: return pd.DataFrame()
        team_stats = unique_matches_by_team.groupby('team_roster').agg(
            Partidas_Jogadas=('won', 'count'), Vitórias=('won', 'sum'),
            Saldo_de_Rounds=('round_diff', 'sum')
        )
        team_stats['Derrotas'] = team_stats['Partidas_Jogadas'] - team_stats['Vitórias']
        team_stats['% de Vitória'] = (team_stats['Vitórias'] / team_stats['Partidas_Jogadas']) * 100
        best_maps = {}
        for roster, group in unique_matches_by_team.groupby('team_roster'):
            map_perf = group.groupby('map')['won'].agg(['count', 'sum'])
            map_perf['win_rate'] = (map_perf['sum'] / map_perf['count']) * 100
            map_perf = map_perf.sort_values(by=['win_rate', 'count'], ascending=[False, False])
            if not map_perf.empty:
                best_map_name, best_map_stats = map_perf.index[0], map_perf.iloc[0]
                best_maps[roster] = f"{best_map_name} ({best_map_stats['win_rate']:.0f}% em {int(best_map_stats['count'])} jogos)"
            else: best_maps[roster] = "N/A"
        team_stats['Melhor Mapa'] = team_stats.index.map(best_maps)
        team_stats['Time'] = team_stats.index.map(lambda roster: ', '.join(sorted(list(roster))))
        final_cols = ['Time', 'Partidas_Jogadas', 'Vitórias', 'Derrotas', '% de Vitória', 'Saldo_de_Rounds', 'Melhor Mapa']
        return team_stats[final_cols].sort_values(by='Partidas_Jogadas', ascending=False).set_index('Time')

    def get_h2h_leaderboard(self, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        if df.empty: return pd.DataFrame()
        match_rosters = df.groupby('match_id')['team_roster'].unique()
        h2h_counts = {}
        for match_id, teams in match_rosters.items():
            if len(teams) == 2:
                team_a, team_b = teams[0], teams[1]
                for p1 in team_a:
                    for p2 in team_b:
                        rivalry = frozenset([p1, p2])
                        h2h_counts[rivalry] = h2h_counts.get(rivalry, 0) + 1
        if not h2h_counts: return pd.DataFrame()
        leaderboard = pd.DataFrame([{'Jogadores': f'{list(r)[0]} vs {list(r)[1]}', 'Confrontos': count} for r, count in h2h_counts.items()])
        return leaderboard.sort_values(by='Confrontos', ascending=False).reset_index(drop=True)

    def get_h2h_cumulative_trend(self, player1, player2, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        p1_rows = df[df['player'] == player1]
        match_ids = p1_rows[p1_rows['opponents'].apply(lambda o: player2 in o)]['match_id'].unique()
        if len(match_ids) == 0: return pd.DataFrame(), pd.DataFrame()
        h2h_df = df[df['match_id'].isin(match_ids)].sort_values(by='date')
        results = {}
        for player in [player1, player2]:
            player_df = h2h_df[h2h_df['player'] == player]
            match_stats = player_df.groupby('match_id').agg(k=('k', 'first'), d=('d', 'first'), won=('won', 'first')).reindex(match_ids).dropna()
            if match_stats.empty: continue
            match_stats['K_Cumulativo'] = match_stats['k'].cumsum()
            match_stats['D_Cumulativo'] = match_stats['d'].cumsum()
            match_stats['Taxa K/D Cumulativa'] = (match_stats['K_Cumulativo'] / match_stats['D_Cumulativo']).fillna(0)
            match_stats['Vitórias_Cumulativas'] = match_stats['won'].cumsum()
            match_stats['Partidas_Totais'] = range(1, len(match_stats) + 1)
            match_stats['% de Vitória Cumulativa'] = (match_stats['Vitórias_Cumulativas'] / match_stats['Partidas_Totais']) * 100
            results[player] = match_stats
        return results.get(player1, pd.DataFrame()), results.get(player2, pd.DataFrame())

    def get_player_evolution_ranking(self, start_date, end_date, min_games=10):
        df = self._filter_by_date(start_date, end_date)
        players = df['player'].unique()
        evolution_data = []
        for player in players:
            player_df = df[df['player'] == player].sort_values(by='date')
            match_stats = player_df.groupby('match_id').agg(k=('k', 'first'), d=('d', 'first'), won=('won', 'first')).reset_index()
            if len(match_stats) < min_games: continue
            match_stats['kd_ratio'] = (match_stats['k'] / match_stats['d']).replace([np.inf, -np.inf], 1).fillna(1)
            X = np.arange(len(match_stats)).reshape(-1, 1)
            model_kd = LinearRegression().fit(X, match_stats['kd_ratio'])
            kd_slope_overall = model_kd.coef_[0]
            last_10 = match_stats.tail(10)
            if len(last_10) >= 2: # Precisa de pelo menos 2 pontos para uma linha
                X_10 = np.arange(len(last_10)).reshape(-1, 1)
                model_kd_10 = LinearRegression().fit(X_10, last_10['kd_ratio'])
                kd_slope_last_10 = model_kd_10.coef_[0]
            else: kd_slope_last_10 = 0.0
            evolution_data.append({'Jogador': player, 'Partidas': len(match_stats), 'Tendência K/D (Geral)': kd_slope_overall, 'Tendência K/D (Últimos 10)': kd_slope_last_10})
        if not evolution_data: return pd.DataFrame(), pd.DataFrame()
        evo_df = pd.DataFrame(evolution_data)
        geral_df = evo_df.sort_values(by='Tendência K/D (Geral)', ascending=False)
        recente_df = evo_df.sort_values(by='Tendência K/D (Últimos 10)', ascending=False)
        return geral_df, recente_df

# --- FUNÇÕES DE PLOTAGEM ---
def create_top_kd_chart(df):
    MIN_GAMES = 5
    filtered_df = df[df['Partidas Jogadas'] >= MIN_GAMES].nlargest(10, 'Taxa K/D')
    if filtered_df.empty: return None, f"Nenhum jogador com {MIN_GAMES} ou mais partidas."
    fig, ax = plt.subplots(figsize=(10, 6)); players = filtered_df.index; kd_ratio = filtered_df['Taxa K/D']
    ax.barh(players, kd_ratio, color='skyblue'); min_val, max_val = kd_ratio.min(), kd_ratio.max()
    ax.set_xlim(left=max(0, min_val * 0.9), right=max_val * 1.1)
    for index, value in enumerate(kd_ratio):
        ax.text(value, index, f' {value:.2f}', va='center')
    ax.set_title(f'Top {len(filtered_df)} Jogadores por Taxa K/D (mín. {MIN_GAMES} partidas)'); ax.set_xlabel('Taxa K/D')
    plt.gca().invert_yaxis(); plt.tight_layout(); buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(fig)
    return buf, None

def create_h2h_trend_chart(trend_p1, trend_p2, player1, player2):
    if trend_p1.empty or trend_p2.empty: return None
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    match_index = range(1, len(trend_p1) + 1)
    ax1.plot(match_index, trend_p1['Taxa K/D Cumulativa'], label=f'K/D {player1}', color='cornflowerblue', marker='o'); ax1.plot(match_index, trend_p2['Taxa K/D Cumulativa'], label=f'K/D {player2}', color='lightcoral', marker='x')
    ax1.set_ylabel('Taxa K/D Cumulativa (nos confrontos)'); ax1.set_title('Evolução do K/D em Confrontos Diretos'); ax1.legend(); ax1.grid(True, linestyle='--')
    ax2.plot(match_index, trend_p1['% de Vitória Cumulativa'], label=f'% Vit. {player1}', color='cornflowerblue', marker='o'); ax2.plot(match_index, trend_p2['% de Vitória Cumulativa'], label=f'% Vit. {player2}', color='lightcoral', marker='x')
    ax2.set_xlabel('Número do Confronto Direto'); ax2.set_ylabel('% de Vitória Cumulativa (nos confrontos)'); ax2.set_title('Evolução da % de Vitória em Confrontos Diretos')
    ax2.legend(); ax2.grid(True, linestyle='--'); fig.tight_layout(); buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(fig); return buf
# ... (outras funções de plotagem existentes)

# --- INTERFACE GRÁFICA (Streamlit) ---
st.set_page_config(layout="wide", page_title="Análise de Partidas CS2")
st.title("📊 Painel de Análise de Partidas de Counter-Strike 2")
if 'team_players' not in st.session_state: st.session_state.team_players = [""] * 5
uploaded_file = st.file_uploader("Carregue seu arquivo 'match_data.txt'", type="txt")

if uploaded_file:
    try:
        string_data = uploaded_file.getvalue().decode("utf-8")
        match_data = [json.loads(line) for line in string_data.splitlines() if line.strip()]
        analyzer = StatsAnalyzer(match_data)
        player_list, map_list = analyzer.get_player_list(), analyzer.get_map_list()
        min_date, max_date = analyzer.df['date'].min().date(), analyzer.df['date'].max().date()
        st.sidebar.header("Filtros")
        date_range = st.sidebar.date_input("Intervalo de Datas:", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        start_date, end_date = (date_range[0], date_range[1]) if len(date_range) == 2 else (min_date, max_date)
        st.sidebar.markdown("---")
        analysis_type = st.sidebar.radio("Tipo de Análise:", ("Visão Geral", "Análise de Tendências", "Melhores Times", "Maiores Rivalidades", "Estatísticas Gerais", "Análise de Jogador", "Análise por Mapa", "Confronto 1x1", "Montar Time"))

        if analysis_type == "Visão Geral":
            # ... (código existente)
            pass

        elif analysis_type == "Análise de Tendências":
            st.header(f"Análise de Tendências ({start_date} a {end_date})")
            st.info("Esta análise mede a 'inclinação' da performance (K/D) de cada jogador ao longo do tempo. Um valor positivo indica melhora, um negativo indica piora.")
            min_games_trend = st.slider("Analisar jogadores com no mínimo quantas partidas?", 5, 30, 10)
            geral_df, recente_df = analyzer.get_player_evolution_ranking(start_date, end_date, min_games=min_games_trend)
            if geral_df.empty:
                st.warning(f"Não há jogadores com {min_games_trend} ou mais partidas para uma análise de tendência.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📈 Maiores Evoluções (Geral)")
                    st.dataframe(geral_df[['Jogador', 'Partidas', 'Tendência K/D (Geral)']].head(10).style.format({'Tendência K/D (Geral)': '{:+.4f}'}))
                with col2:
                    st.subheader("📉 Maiores Quedas (Geral)")
                    st.dataframe(geral_df[['Jogador', 'Partidas', 'Tendência K/D (Geral)']].tail(10).sort_values(by='Tendência K/D (Geral)').style.format({'Tendência K/D (Geral)': '{:+.4f}'}))
                st.markdown("---")
                col3, col4 = st.columns(2)
                with col3:
                    st.subheader("🔥 Em Alta (Últimos 10 Jogos)")
                    st.dataframe(recente_df[['Jogador', 'Partidas', 'Tendência K/D (Últimos 10)']].head(10).style.format({'Tendência K/D (Últimos 10)': '{:+.4f}'}))
                with col4:
                    st.subheader("❄️ Em Baixa (Últimos 10 Jogos)")
                    st.dataframe(recente_df[['Jogador', 'Partidas', 'Tendência K/D (Últimos 10)']].tail(10).sort_values(by='Tendência K/D (Últimos 10)').style.format({'Tendência K/D (Últimos 10)': '{:+.4f}'}))

        elif analysis_type == "Melhores Times":
            # ... (código existente)
            pass
        
        elif analysis_type == "Maiores Rivalidades":
            st.header(f"Maiores Rivalidades ({start_date} a {end_date})")
            h2h_leaderboard = analyzer.get_h2h_leaderboard(start_date, end_date)
            if not h2h_leaderboard.empty:
                st.dataframe(h2h_leaderboard, use_container_width=True)
            else:
                st.info("Nenhum confronto direto encontrado no período.")
        
        elif analysis_type == "Confronto 1x1":
            # ... (UI existente)
            if player1 and player2:
                # ...
                if h2h_overall:
                    st.subheader("Gráfico Comparativo Geral")
                    st.image(create_h2h_comparison_chart(h2h_overall, player1, player2))
                    st.subheader("Evolução da Performance nos Confrontos")
                    trend_p1, trend_p2 = analyzer.get_h2h_cumulative_trend(player1, player2, start_date, end_date)
                    h2h_trend_chart = create_h2h_trend_chart(trend_p1, trend_p2, player1, player2)
                    if h2h_trend_chart: st.image(h2h_trend_chart)
                    # ... (resto da UI H2H)
        
        # ... (restante do código da UI)
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}"); import traceback; st.error(traceback.format_exc())
else:
    st.info("Aguardando o upload do arquivo `match_data.txt` para iniciar a análise.")
