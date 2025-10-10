import streamlit as st
import pandas as pd
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import io

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
        self.rosters_by_match = self.df.groupby('match_id').agg(
            team_roster=('team_roster', 'first'),
            opponents=('opponents', 'first'),
            date=('date', 'first'),
            map=('map', 'first'),
            rounds_ganhos=('rounds_ganhos', 'first'),
            rounds_perdidos=('rounds_perdidos', 'first'),
            won=('won', 'first')
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
        # Implementação existente...
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

        teammate_records = []
        for match_id, row in player_matches.groupby('match_id').first().iterrows():
            for teammate in row['team_roster']:
                if teammate != player_name:
                    teammate_records.append({'match_id': match_id, 'teammate': teammate, 'won': row['won']})
        
        if not teammate_records: return pd.DataFrame()

        teammate_df = pd.DataFrame(teammate_records)
        stats = teammate_df.groupby('teammate').agg(
            Partidas_Juntos=('match_id', 'nunique'),
            Vitórias=('won', 'sum')
        )
        stats['% de Vitória Juntos'] = (stats['Vitórias'] / stats['Partidas_Juntos']) * 100
        return stats.sort_values(by='% de Vitória Juntos', ascending=not best).head(5)

    def get_core_player_stats(self, core_players, start_date, end_date):
        # Implementação existente...
        df = self._filter_by_date(start_date, end_date)
        all_players = sorted(df['player'].unique())
        if not core_players:
            return {"Partidas Juntos": 0, "% de Vitória": 0}, all_players, pd.DataFrame()
        
        core_set = set(core_players)
        relevant_rosters = self.rosters_by_match[self.rosters_by_match.index.isin(df['match_id'].unique())]
        valid_match_ids = [mid for mid, roster in relevant_rosters.items() if core_set.issubset(roster['team_roster'])]
        
        if not valid_match_ids:
            return {"Partidas Juntos": 0, "% de Vitória": 0}, [], pd.DataFrame()

        match_info = self.rosters_by_match.loc[valid_match_ids]
        total_games = len(match_info)
        wins = int(match_info['won'].sum())
        stats = {"Partidas Juntos": total_games, "Vitórias": wins, "Derrotas": total_games-wins, "% de Vitória": (wins / total_games * 100) if total_games > 0 else 0}

        possible_next_players = set()
        for roster in match_info['team_roster']:
            possible_next_players.update(roster)
        
        # Histórico de partidas
        history = match_info.copy()
        history['Oponentes'] = history['opponents'].apply(lambda x: ', '.join(sorted(list(x))))
        history['Placar'] = history.apply(lambda row: f"{row['rounds_ganhos']} x {row['rounds_perdidos']}" if row['won'] else f"{row['rounds_perdidos']} x {row['rounds_ganhos']}", axis=1)
        history.rename(columns={'map': 'Mapa', 'date': 'Data'}, inplace=True)
        
        return stats, sorted(list(possible_next_players - core_set)), history[['Data', 'Mapa', 'Placar', 'Oponentes']]

    def get_match_count_over_time(self, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        if df.empty: return pd.DataFrame()
        unique_matches = df.groupby('match_id')['date'].first().reset_index()
        return unique_matches.groupby(unique_matches['date'].dt.date)['match_id'].count()

# --- FUNÇÕES DE PLOTAGEM ---
def create_match_history_chart(series):
    if series.empty: return None
    fig, ax = plt.subplots(figsize=(12, 6))
    series.plot(kind='bar', ax=ax, color='teal')
    ax.set_title('Histórico de Partidas por Dia', fontsize=16)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Número de Partidas', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(fig); return buf

# ... Suas outras funções de plotagem (create_player_trend_chart, create_h2h_comparison_chart) ...

# --- INTERFACE GRÁFICA (Streamlit) ---
st.set_page_config(layout="wide", page_title="Análise de Partidas CS2")
st.title("📊 Painel de Análise de Partidas de Counter-Strike 2")

if 'team_players' not in st.session_state:
    st.session_state.team_players = [""] * 5

uploaded_file = st.file_uploader("Carregue seu arquivo 'match_data.txt'", type="txt")

if uploaded_file:
    try:
        string_data = uploaded_file.getvalue().decode("utf-8")
        match_data = [json.loads(line) for line in string_data.splitlines() if line.strip()]
        analyzer = StatsAnalyzer(match_data)
        player_list = analyzer.get_player_list()
        map_list = analyzer.get_map_list()
        min_date, max_date = analyzer.df['date'].min().date(), analyzer.df['date'].max().date()
        
        st.sidebar.header("Filtros")
        date_range = st.sidebar.date_input("Intervalo de Datas:", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        start_date, end_date = (date_range[0], date_range[1]) if len(date_range) == 2 else (min_date, max_date)
        
        st.sidebar.markdown("---")
        analysis_type = st.sidebar.radio("Tipo de Análise:", ("Visão Geral", "Estatísticas Gerais", "Análise de Jogador", "Confronto 1x1", "Análise por Mapa", "Montar Time"))

        if analysis_type == "Visão Geral":
            st.header(f"Visão Geral das Partidas ({start_date} a {end_date})")
            match_counts = analyzer.get_match_count_over_time(start_date, end_date)
            history_chart = create_match_history_chart(match_counts)
            if history_chart: st.image(history_chart)
            else: st.info("Nenhuma partida encontrada no período para gerar o gráfico.")

        elif analysis_type == "Estatísticas Gerais":
            st.header(f"Estatísticas Gerais de Todos os Jogadores ({start_date} a {end_date})")
            # Adicione sua lógica de estatísticas gerais aqui, se desejar, como o gráfico de Top K/D
            sort_option = st.selectbox("Ordenar por:", ['Partidas Jogadas', '% de Vitória', 'Taxa K/D', 'Saldo de Rounds', 'Abates (K)'])
            stats_df = analyzer.get_overall_player_stats(start_date, end_date, sort_by=sort_option)
            st.dataframe(stats_df.style.format({'% de Vitória': '{:.2f}%', 'Taxa K/D': '{:.2f}'}))

        elif analysis_type == "Análise de Jogador":
            player_name = st.sidebar.selectbox("Selecione o Jogador:", player_list)
            if player_name:
                st.header(f"Análise Individual de {player_name}")
                summary = analyzer.get_player_overall_stats_summary(player_name, start_date, end_date)
                if summary:
                    # Métricas e Gráfico de Tendência...
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Partidas", summary["Partidas Jogadas"]); c2.metric("Vitórias", summary["Vitórias"])
                    c3.metric("Derrotas", summary["Derrotas"]); c4.metric("% de Vitória", summary["% de Vitória"])
                    
                    st.subheader("Análise de Companheiros")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Melhores Companheiros (% de Vitória)**")
                        best_teammates = analyzer.get_performance_with_teammates(player_name, start_date, end_date, best=True)
                        st.dataframe(best_teammates.style.format({'% de Vitória Juntos': '{:.2f}%'}))
                    with col2:
                        st.write("**Piores Companheiros (% de Vitória)**")
                        worst_teammates = analyzer.get_performance_with_teammates(player_name, start_date, end_date, best=False)
                        st.dataframe(worst_teammates.style.format({'% de Vitória Juntos': '{:.2f}%'}))

        elif analysis_type == "Análise por Mapa":
            selected_map = st.sidebar.selectbox("Selecione um Mapa:", ["Todos"] + map_list)
            st.header(f"Análise por Mapa: {selected_map}")
            # ... Implementar a lógica para exibir estatísticas do mapa
            
        elif analysis_type == "Montar Time":
            st.header("Montar Time (Seleção em Cascata)")
            st.sidebar.subheader("Selecione os Jogadores")
            # ... Lógica do seletor em cascata ...
            if st.sidebar.button("Limpar Time"):
                st.session_state.team_players = [""] * 5
                st.experimental_rerun()

            selected_so_far = []
            options = player_list
            for i in range(5):
                is_disabled = (i > 0 and not st.session_state.team_players[i-1])
                if i > 0 and st.session_state.team_players[i-1]:
                    selected_so_far = [p for p in st.session_state.team_players if p]
                    _, options = analyzer.get_core_player_stats(selected_so_far, start_date, end_date)
                
                current_player = st.session_state.team_players[i]
                if current_player and current_player not in options:
                    options = [current_player] + options
                
                st.session_state.team_players[i] = st.sidebar.selectbox(f"Jogador {i+1}", [""] + options, index=([""] + options).index(current_player) if current_player in options else 0, key=f'player_{i}', disabled=is_disabled)

            final_team = [p for p in st.session_state.team_players if p]
            if len(final_team) >= 2:
                st.subheader("Estatísticas do Núcleo Selecionado")
                team_name = " & ".join(sorted(final_team))
                st.write(f"#### {team_name}")
                
                core_stats, _, history_df = analyzer.get_core_player_stats(final_team, start_date, end_date)
                if core_stats["Partidas Juntos"] > 0:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Partidas Juntos", core_stats["Partidas Juntos"])
                    c2.metric("Vitórias", core_stats["Vitórias"])
                    c3.metric("Derrotas", core_stats["Derrotas"])
                    c4.metric("% de Vitória", f"{core_stats['% de Vitória']:.2f}%")
                    
                    if len(final_team) == 5:
                        st.subheader("Histórico de Partidas do Time")
                        st.dataframe(history_df)
                else:
                    st.warning("Este núcleo de jogadores nunca jogou junto.")
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        import traceback
        st.error(traceback.format_exc())
else:
    st.info("Aguardando o upload do arquivo `match_data.txt` para iniciar a análise.")
