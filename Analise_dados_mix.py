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

            for team_players, winner, score_diff, r_ganhos, r_perdidos, opponents in [(team_a_players, is_team_a_winner, score_a - score_b, score_a, score_b, team_b_players), (team_b_players, is_team_a_winner is not None and not is_team_a_winner, score_b - score_a, score_b, score_a, team_a_players)]:
                original_team_data = match.get('team_a') if team_players == team_a_players else match.get('team_b')
                for player_stats in original_team_data:
                    player_name = self._unify_player_names(player_stats.get('player'))
                    records.append({
                        'match_id': match_id, 'date': match.get('date'), 'map': match.get('map'), 'player': player_name,
                        'k': player_stats.get('k', 0), 'a': player_stats.get('a', 0), 'd': player_stats.get('d', 0),
                        'team_roster': frozenset(team_players), 'won': winner, 'round_diff': score_diff,
                        'rounds_ganhos': r_ganhos, 'rounds_perdidos': r_perdidos,
                        'teammates': team_players - {player_name}, 'opponents': opponents
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

    # --- MÉTODOS DE ANÁLISE GERAL ---
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
            "Rounds Ganhos": int(player_df['rounds_ganhos'].sum() / player_df.groupby('match_id')['player'].count().mean()),
            "Rounds Perdidos": int(player_df['rounds_perdidos'].sum() / player_df.groupby('match_id')['player'].count().mean()),
            "Saldo de Rounds": int(match_stats['round_diff'].sum())
        }
        
    def get_map_leaderboard(self, map_name, start_date, end_date, sort_by='% de Vitória'):
        df = self._filter_by_date(start_date, end_date)
        map_df = df[df['map'] == map_name]
        if map_df.empty: return pd.DataFrame()

        temp_analyzer = type(self)([]) 
        temp_analyzer.df = map_df
        return temp_analyzer.get_overall_player_stats(start_date, end_date, sort_by)

    def get_duo_stats(self, player1, player2, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        core_set = {player1, player2}
        
        valid_match_ids = [
            match_id for match_id, roster in self.rosters_by_match.items() 
            if core_set.issubset(roster) and self.rosters_by_match.index.isin(df['match_id'].unique()).any()
        ]
        if not valid_match_ids: return None

        final_df = df[df['match_id'].isin(valid_match_ids)]
        match_info = final_df.groupby('match_id').first()
        total_games = len(match_info)
        wins = int(match_info['won'].sum())
        
        combined_stats = final_df[final_df['player'].isin(core_set)].groupby('match_id')[['k', 'd']].sum()
        total_k = combined_stats['k'].sum(); total_d = combined_stats['d'].sum()
        
        return {
            "Partidas Juntos": total_games, "Vitórias": wins, "Derrotas": total_games - wins,
            "% de Vitória da Dupla": f"{(wins/total_games*100):.2f}%",
            "K/D Combinado": f"{(total_k/total_d):.2f}" if total_d > 0 else "N/A"
        }
    
    def get_h2h_overall(self, player1, player2, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        p1_rows = df[df['player'] == player1]
        match_ids = p1_rows[p1_rows['opponents'].apply(lambda o: player2 in o)]['match_id'].unique()
        if len(match_ids) == 0: return None
        
        h2h_df = df[df['match_id'].isin(match_ids)]
        p1_stats = h2h_df[h2h_df['player'] == player1]; p2_stats = h2h_df[h2h_df['player'] == player2]
        
        p1_wins = p1_stats.groupby('match_id')['won'].first().sum()
        
        return {
            "Partidas H2H": len(match_ids), f"Vitórias {player1}": int(p1_wins), f"Vitórias {player2}": len(match_ids) - int(p1_wins),
            f"Kills {player1}": p1_stats['k'].sum(), f"Deaths {player1}": p1_stats['d'].sum(),
            f"Taxa K/D {player1}": (p1_stats['k'].sum() / p1_stats['d'].sum()) if p1_stats['d'].sum() > 0 else 0,
            f"Kills {player2}": p2_stats['k'].sum(), f"Deaths {player2}": p2_stats['d'].sum(),
            f"Taxa K/D {player2}": (p2_stats['k'].sum() / p2_stats['d'].sum()) if p2_stats['d'].sum() > 0 else 0
        }

    # --- NOVA LÓGICA PARA ANÁLISE DE TIME DINÂMICA ---
    def get_core_player_stats(self, core_players, start_date, end_date):
        df = self._filter_by_date(start_date, end_date)
        all_players = sorted(df['player'].unique())
        if not core_players:
            return {"Partidas Juntos": 0, "% de Vitória": 0}, all_players
        
        core_set = set(core_players)
        
        relevant_rosters = self.rosters_by_match[self.rosters_by_match.index.isin(df['match_id'].unique())]
        valid_match_ids = [
            match_id for match_id, roster in relevant_rosters.items() 
            if core_set.issubset(roster)
        ]
        
        if not valid_match_ids:
            return {"Partidas Juntos": 0, "% de Vitória": 0}, []

        core_df = df[df['match_id'].isin(valid_match_ids)]
        match_info = core_df.groupby('match_id').first()
        total_games = len(match_info)
        wins = int(match_info['won'].sum())
        win_rate = (wins / total_games * 100) if total_games > 0 else 0
        stats = {"Partidas Juntos": total_games, "% de Vitória": win_rate}

        possible_next_players = set()
        for roster in relevant_rosters[relevant_rosters.index.isin(valid_match_ids)]:
            possible_next_players.update(roster)
        
        valid_next_players = sorted(list(possible_next_players - core_set))
        return stats, valid_next_players

# --- FUNÇÕES DE PLOTAGEM (sem alterações, apenas adicione as suas aqui) ---
def create_h2h_comparison_chart(h2h_stats, player1, player2):
    metrics = ['Vitórias', 'Kills', 'Deaths', 'Taxa K/D']
    p1_values = [h2h_stats[f'Vitórias {player1}'], h2h_stats[f'Kills {player1}'], h2h_stats[f'Deaths {player1}'], h2h_stats[f'Taxa K/D {player1}']]
    p2_values = [h2h_stats[f'Vitórias {player2}'], h2h_stats[f'Kills {player2}'], h2h_stats[f'Deaths {player2}'], h2h_stats[f'Taxa K/D {player2}']]
    x, width = np.arange(len(metrics)), 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, p1_values, width, label=player1, color='cornflowerblue')
    rects2 = ax.bar(x + width/2, p2_values, width, label=player2, color='lightcoral')
    ax.set_ylabel('Valores'); ax.set_title(f'Comparativo H2H: {player1} vs {player2}'); ax.set_xticks(x); ax.set_xticklabels(metrics); ax.legend()
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}' if isinstance(height, float) and height < 5 else f'{int(height)}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    fig.tight_layout(); buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(fig); return buf


# --- INTERFACE GRÁFICA (Streamlit) ---
st.set_page_config(layout="wide", page_title="Análise de Partidas CS2")
st.title("📊 Painel de Análise de Partidas de Counter-Strike 2")

if 'selected_players' not in st.session_state:
    st.session_state.selected_players = []

uploaded_file = st.file_uploader("Carregue seu arquivo 'match_data.txt'", type="txt")

if uploaded_file:
    try:
        string_data = uploaded_file.getvalue().decode("utf-8")
        match_data = [json.loads(line) for line in string_data.splitlines() if line.strip()]
        analyzer = StatsAnalyzer(match_data)
        player_list = analyzer.get_player_list()
        map_list = analyzer.get_map_list()
        min_date, max_date = analyzer.df['date'].min().date(), analyzer.df['date'].max().date()
        
        st.sidebar.header("Filtros e Seleção")
        date_range = st.sidebar.date_input("Selecione o Intervalo de Datas:", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        start_date, end_date = (date_range[0], date_range[1]) if len(date_range) == 2 else (min_date, max_date)
        
        st.sidebar.markdown("---")
        analysis_type = st.sidebar.radio("Tipo de Análise:", ("Estatísticas Gerais", "Análise de Jogador", "Análise de Dupla (Juntos)", "Análise de Time (Estratégica)", "Confronto 1x1 (Contra)", "Análise por Mapa"))

        if analysis_type == "Estatísticas Gerais":
            st.header(f"Estatísticas Gerais ({start_date} a {end_date})")
            sort_option = st.selectbox("Ordenar por:", ['Partidas Jogadas', '% de Vitória', 'Taxa K/D', 'Saldo de Rounds', 'Abates (K)'])
            stats_df = analyzer.get_overall_player_stats(start_date, end_date, sort_by=sort_option)
            if not stats_df.empty: st.dataframe(stats_df.style.format({'% de Vitória': '{:.2f}%', 'Taxa K/D': '{:.2f}'}))
            else: st.info("Nenhuma partida encontrada no período.")

        elif analysis_type == "Análise de Jogador":
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
                else: st.warning("Nenhum dado para este jogador no período.")

        elif analysis_type == "Análise de Dupla (Juntos)":
            st.sidebar.subheader("Selecione a Dupla")
            player1 = st.sidebar.selectbox("Jogador 1:", player_list, key='duo_p1')
            player2 = st.sidebar.selectbox("Jogador 2:", [p for p in player_list if p != player1], key='duo_p2')
            if player1 and player2 and player1 != player2:
                st.header(f"Análise da Dupla: {player1} & {player2}")
                duo_stats = analyzer.get_duo_stats(player1, player2, start_date, end_date)
                if duo_stats:
                    c1,c2,c3,c4,c5 = st.columns(5)
                    c1.metric("Partidas Juntos", duo_stats["Partidas Juntos"]); c2.metric("Vitórias", duo_stats["Vitórias"])
                    c3.metric("Derrotas", duo_stats["Derrotas"]); c4.metric("% de Vitória", duo_stats["% de Vitória da Dupla"])
                    c5.metric("K/D Combinado", duo_stats["K/D Combinado"])
                else: st.warning("Esta dupla nunca jogou junta no período selecionado.")

        elif analysis_type == "Análise de Time (Estratégica)":
            st.header("Análise de Time Estratégica")
            st.sidebar.subheader("Monte seu Time")

            stats, next_options = analyzer.get_core_player_stats(st.session_state.selected_players, start_date, end_date)
            
            # Opções para o multiselect: jogadores já selecionados + próximas opções válidas
            multiselect_options = sorted(list(set(st.session_state.selected_players) | set(next_options)))

            # Usando um formulário para evitar reruns indesejados a cada seleção
            with st.sidebar.form(key='team_selection_form'):
                st.session_state.selected_players = st.multiselect(
                    "Selecione de 1 a 5 jogadores:",
                    multiselect_options,
                    default=st.session_state.selected_players,
                    max_selections=5,
                    key='player_multiselect'
                )
                submit_button = st.form_submit_button(label='Analisar Time')

            if submit_button and not st.session_state.selected_players:
                 st.sidebar.warning("Selecione pelo menos um jogador.")

            st.subheader("Time Selecionado")
            if not st.session_state.selected_players:
                st.info("Comece selecionando jogadores na barra lateral e clique em 'Analisar Time'.")
            else:
                team_name = " & ".join(sorted(st.session_state.selected_players))
                st.write(f"#### {team_name}")
                if len(st.session_state.selected_players) > 1:
                    if stats["Partidas Juntos"] > 0:
                        c1, c2 = st.columns(2)
                        c1.metric("Partidas com este núcleo", stats["Partidas Juntos"])
                        c2.metric("% de Vitória", f"{stats['% de Vitória']:.2f}%")
                    else:
                        st.warning("Este núcleo de jogadores nunca jogou junto.")
                else:
                    st.info("Selecione mais um jogador para ver as estatísticas do núcleo.")

        elif analysis_type == "Confronto 1x1 (Contra)":
            st.sidebar.subheader("Selecione os Jogadores")
            player1 = st.sidebar.selectbox("Jogador 1:", player_list, key='h2h_p1')
            player2 = st.sidebar.selectbox("Jogador 2:", [p for p in player_list if p != player1], key='h2h_p2')
            if player1 and player2:
                st.header(f"Confronto Direto: {player1} vs {player2}")
                h2h_overall = analyzer.get_h2h_overall(player1, player2, start_date, end_date)
                if h2h_overall:
                    st.subheader("Gráfico Comparativo Geral")
                    h2h_chart = create_h2h_comparison_chart(h2h_overall, player1, player2)
                    st.image(h2h_chart)
                else: st.warning("Estes jogadores nunca se enfrentaram no período selecionado.")

        elif analysis_type == "Análise por Mapa":
            selected_map = st.sidebar.selectbox("Selecione um Mapa:", map_list)
            st.header(f"Estatísticas do Mapa: {selected_map}")
            sort_option_map = st.selectbox("Ordenar por:", ['Partidas Jogadas', '% de Vitória', 'Taxa K/D'], key='map_sort')
            map_stats = analyzer.get_map_leaderboard(selected_map, start_date, end_date, sort_by=sort_option_map)
            if not map_stats.empty: st.dataframe(map_stats.style.format({'% de Vitória': '{:.2f}%', 'Taxa K/D': '{:.2f}'}))
            else: st.info("Nenhuma partida neste mapa no período.")

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo. Detalhe: {e}")
        st.error("Se o erro persistir, verifique se o formato do seu 'match_data.txt' está correto.")
else:
    st.info("Aguardando o upload do arquivo `match_data.txt` para iniciar a análise.")
