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
            score_a = match.get('score_a', 0); score_b = match.get('score_b', 0)
            is_team_a_winner = None
            if score_a > score_b: is_team_a_winner = True
            elif score_b > score_a: is_team_a_winner = False
            
            team_a_players = {self._unify_player_names(p.get('player')) for p in match.get('team_a', [])}
            team_b_players = {self._unify_player_names(p.get('player')) for p in match.get('team_b', [])}

            for team_players, winner, score_diff, r_ganhos, r_perdidos in [(team_a_players, is_team_a_winner, score_a - score_b, score_a, score_b), (team_b_players, is_team_a_winner is not None and not is_team_a_winner, score_b - score_a, score_b, score_a)]:
                original_team_data = match.get('team_a') if team_players == team_a_players else match.get('team_b')
                for player_stats in original_team_data:
                    player_name = self._unify_player_names(player_stats.get('player'))
                    records.append({
                        'match_id': match_id, 'date': match.get('date'), 'map': match.get('map'), 'player': player_name,
                        'k': player_stats.get('k', 0), 'a': player_stats.get('a', 0), 'd': player_stats.get('d', 0),
                        'team_roster': frozenset(team_players), 'won': winner, 'round_diff': score_diff,
                        'rounds_ganhos': r_ganhos, 'rounds_perdidos': r_perdidos
                    })
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def _precompute_rosters(self):
        self.rosters_by_match = self.df.groupby('match_id')['team_roster'].first()

    def _filter_by_date(self, start_date, end_date):
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())
        return self.df[(self.df['date'] >= start_dt) & (self.df['date'] <= end_dt)]

    def get_player_list(self): return sorted(self.df['player'].unique())
    def get_map_list(self): return sorted(self.df['map'].unique())

    # --- MÉTODOS DE ANÁLISE RESTAURADOS E CORRIGIDOS ---
    def get_overall_player_stats(self, start_date, end_date, sort_by='Partidas Jogadas'):
        df = self._filter_by_date(start_date, end_date)
        if df.empty: return pd.DataFrame()
        stats = df.groupby('player').agg(
            **{'Partidas Jogadas': ('match_id', 'nunique'), 'Abates (K)': ('k', 'sum'), 'Assistências (A)': ('a', 'sum'),
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
        total_games = len(match_stats)
        wins = int(match_stats['won'].sum())
        
        return {
            "Partidas Jogadas": total_games, "Vitórias": wins, "Derrotas": total_games - wins,
            "% de Vitória": f"{(wins / total_games * 100):.2f}%" if total_games > 0 else "0.00%",
            "Rounds Ganhos": int(match_stats['rounds_ganhos'].sum()),
            "Rounds Perdidos": int(match_stats['rounds_perdidos'].sum()),
            "Saldo de Rounds": int(match_stats['round_diff'].sum())
        }
        
    def get_map_leaderboard(self, map_name, start_date, end_date, sort_by='% de Vitória'):
        df = self._filter_by_date(start_date, end_date)
        map_df = df[df['map'] == map_name]
        if map_df.empty: return pd.DataFrame()

        # Reutiliza a lógica de get_overall_player_stats no dataframe filtrado
        temp_analyzer = StatsAnalyzer([]) # Cria objeto vazio
        temp_analyzer.df = map_df # Atribui o dataframe filtrado
        return temp_analyzer.get_overall_player_stats(start_date, end_date, sort_by)

    def get_duo_stats(self, player1, player2, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        
        # Encontra as partidas onde ambos jogaram
        p1_matches = set(df[df['player'] == player1]['match_id'])
        p2_matches = set(df[df['player'] == player2]['match_id'])
        common_matches = list(p1_matches.intersection(p2_matches))
        
        if not common_matches: return None

        # Filtra o DF para essas partidas
        duo_df = df[df['match_id'].isin(common_matches)]
        
        # Encontra as partidas em que estavam no mesmo time
        same_team_matches = []
        for match_id, group in duo_df.groupby('match_id'):
            rosters = group['team_roster'].unique()
            for roster in rosters:
                if player1 in roster and player2 in roster:
                    same_team_matches.append(match_id)
                    break
        
        if not same_team_matches: return None

        final_df = df[df['match_id'].isin(same_team_matches)]
        match_info = final_df.groupby('match_id').first()
        total_games = len(match_info)
        wins = int(match_info['won'].sum())
        
        combined_stats = final_df[final_df['player'].isin([player1, player2])].groupby('match_id')[['k', 'd']].sum()
        total_k = combined_stats['k'].sum()
        total_d = combined_stats['d'].sum()
        
        return {
            "Partidas Juntos": total_games, "Vitórias": wins, "Derrotas": total_games - wins,
            "% de Vitória da Dupla": f"{(wins/total_games*100):.2f}%",
            "K/D Combinado": f"{(total_k/total_d):.2f}" if total_d > 0 else "N/A"
        }
        
    def get_h2h_overall(self, player1, player2, start_date, end_date):
        # Implementação existente
        pass

    def get_h2h_by_map(self, player1, player2, start_date, end_date):
        # Implementação existente
        pass
        
    # --- NOVOS MÉTODOS PARA ANÁLISE DINÂMICA DE TIME ---
    def get_core_player_stats(self, core_players, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        if not core_players:
            return {"Partidas Juntos": 0, "% de Vitória": 0}, sorted(df['player'].unique())
        
        core_set = set(core_players)
        
        # Encontra match_ids que contêm todos os jogadores do núcleo no mesmo time
        valid_match_ids = []
        for match_id, roster in self.rosters_by_match.items():
            if core_set.issubset(roster):
                valid_match_ids.append(match_id)
        
        if not valid_match_ids:
            return {"Partidas Juntos": 0, "% de Vitória": 0}, []

        # Calcula as estatísticas
        core_df = df[df['match_id'].isin(valid_match_ids)]
        match_info = core_df.groupby('match_id').first()
        total_games = len(match_info)
        wins = int(match_info['won'].sum())
        win_rate = (wins / total_games * 100) if total_games > 0 else 0

        stats = {"Partidas Juntos": total_games, "% de Vitória": win_rate}

        # Encontra os próximos jogadores possíveis
        possible_next_players = set()
        for roster in self.rosters_by_match[self.rosters_by_match.index.isin(valid_match_ids)]:
            possible_next_players.update(roster)
        
        valid_next_players = sorted(list(possible_next_players - core_set))
        
        return stats, valid_next_players

# --- FUNÇÕES DE PLOTAGEM (sem alterações) ---
# ... (Suas funções create_top_kd_chart, create_player_trend_chart, create_h2h_comparison_chart)

# --- INTERFACE GRÁFICA (Streamlit) ---
st.set_page_config(layout="wide", page_title="Análise de Partidas CS2")
st.title("📊 Painel de Análise de Partidas de Counter-Strike 2")

if 'selected_players' not in st.session_state:
    st.session_state.selected_players = []
    
uploaded_file = st.file_uploader("Carregue seu arquivo 'match_data.txt'", type="txt")

if uploaded_file:
    # ... Lógica de carregamento e inicialização do analyzer ...
    string_data = uploaded_file.getvalue().decode("utf-8")
    match_data = [json.loads(line) for line in string_data.splitlines() if line.strip()]
    analyzer = StatsAnalyzer(match_data)
    player_list = analyzer.get_player_list()
    map_list = analyzer.get_map_list()
    start_date, end_date = analyzer.df['date'].min().date(), analyzer.df['date'].max().date()
    
    st.sidebar.header("Filtros e Seleção")
    date_range = st.sidebar.date_input("Selecione o Intervalo de Datas:", value=(start_date, end_date), min_value=start_date, max_value=end_date)
    start_date, end_date = (date_range[0], date_range[1]) if len(date_range) == 2 else (start_date, end_date)
    
    st.sidebar.markdown("---")
    analysis_type = st.sidebar.radio("Tipo de Análise:", ("Estatísticas Gerais", "Análise por Mapa", "Análise de Jogador", "Análise de Dupla (Juntos)", "Análise de Time (Dinâmica)"))
    
    if analysis_type == "Análise de Jogador":
        player_name = st.sidebar.selectbox("Selecione o Jogador:", player_list)
        if player_name:
            st.header(f"Análise Individual de {player_name}")
            summary = analyzer.get_player_overall_stats_summary(player_name, start_date, end_date)
            if summary:
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Partidas", summary["Partidas Jogadas"]); c2.metric("Vitórias", summary["Vitórias"])
                c3.metric("Derrotas", summary["Derrotas"]); c4.metric("% de Vitória", summary["% de Vitória"])
                st.subheader("Performance de Rounds")
                c5,c6,c7 = st.columns(3)
                c5.metric("Rounds Ganhos", summary["Rounds Ganhos"]); c6.metric("Rounds Perdidos", summary["Rounds Perdidos"])
                c7.metric("Saldo de Rounds", summary["Saldo de Rounds"])
    
    elif analysis_type == "Análise de Dupla (Juntos)":
        st.sidebar.subheader("Selecione a Dupla")
        player1 = st.sidebar.selectbox("Jogador 1:", player_list)
        player2 = st.sidebar.selectbox("Jogador 2:", [p for p in player_list if p != player1])
        if player1 and player2 and player1 != player2:
            st.header(f"Análise da Dupla: {player1} & {player2}")
            duo_stats = analyzer.get_duo_stats(player1, player2, start_date, end_date)
            if duo_stats:
                c1,c2,c3,c4,c5 = st.columns(5)
                c1.metric("Partidas Juntos", duo_stats["Partidas Juntos"]); c2.metric("Vitórias", duo_stats["Vitórias"])
                c3.metric("Derrotas", duo_stats["Derrotas"]); c4.metric("% de Vitória", duo_stats["% de Vitória da Dupla"])
                c5.metric("K/D Combinado", duo_stats["K/D Combinado"])
            else:
                st.warning("Esta dupla nunca jogou junta no período selecionado.")

    elif analysis_type == "Análise de Time (Dinâmica)":
        st.header("Análise de Time Dinâmica")
        st.sidebar.subheader("Monte seu Time")

        if st.sidebar.button("Limpar Seleção"):
            st.session_state.selected_players = []
            st.experimental_rerun()

        # Lógica de seleção em cascata
        current_selection = st.session_state.selected_players
        stats, next_options = analyzer.get_core_player_stats(current_selection, start_date, end_date)
        
        if len(current_selection) < 5:
            next_player = st.sidebar.selectbox(f"Selecione o Jogador {len(current_selection) + 1}:", [""] + next_options, key=f"player_{len(current_selection)}")
            if next_player:
                st.session_state.selected_players.append(next_player)
                st.experimental_rerun()

        st.subheader("Time Selecionado")
        if not current_selection:
            st.info("Comece selecionando o primeiro jogador na barra lateral.")
        else:
            team_name = " & ".join(current_selection)
            st.write(f"#### {team_name}")
            
            # Mostra estatísticas do núcleo atual
            if stats["Partidas Juntos"] > 0:
                c1, c2 = st.columns(2)
                c1.metric("Partidas com este núcleo", stats["Partidas Juntos"])
                c2.metric("% de Vitória", f"{stats['% de Vitória']:.2f}%")
            else:
                st.warning("Este núcleo de jogadores nunca jogou junto.")
    
    # Adicione as outras abas ("Estatísticas Gerais", "Análise por Mapa", etc.) aqui
    # ...

else:
    st.info("Aguardando o upload do arquivo `match_data.txt` para iniciar a análise.")
