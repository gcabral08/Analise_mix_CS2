import streamlit as st
import pandas as pd
import json
from collections import Counter

# --- LÓGICA DE ANÁLISE (CLASSE StatsAnalyzer) ---
# Esta classe permanece a mesma.

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
            if score_a == score_b: is_team_a_winner = None
            else: is_team_a_winner = score_a > score_b
            
            for player_stats in match.get('team_a', []):
                player_name = self._unify_player_names(player_stats.get('player'))
                records.append({'match_id': match_id, 'date': match.get('date'), 'map': match.get('map'),'player': player_name, 'k': player_stats.get('k', 0), 'a': player_stats.get('a', 0), 'd': player_stats.get('d', 0),'team': 'A', 'won': is_team_a_winner, 'round_diff': score_a - score_b,'teammates': [self._unify_player_names(p.get('player')) for p in match.get('team_a', []) if p.get('player') != player_stats.get('player')],'opponents': [self._unify_player_names(p.get('player')) for p in match.get('team_b', [])]})
            for player_stats in match.get('team_b', []):
                player_name = self._unify_player_names(player_stats.get('player'))
                records.append({'match_id': match_id, 'date': match.get('date'), 'map': match.get('map'),'player': player_name, 'k': player_stats.get('k', 0), 'a': player_stats.get('a', 0), 'd': player_stats.get('d', 0),'team': 'B', 'won': is_team_a_winner is not None and not is_team_a_winner, 'round_diff': score_b - score_a,'teammates': [self._unify_player_names(p.get('player')) for p in match.get('team_b', []) if p.get('player') != player_stats.get('player')],'opponents': [self._unify_player_names(p.get('player')) for p in match.get('team_a', [])]})
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def get_player_list(self):
        return sorted(self.df['player'].unique())

    def get_map_list(self):
        return sorted(self.df['map'].unique())

    def get_overall_player_stats(self, sort_by='Partidas Jogadas'):
        stats = self.df.groupby('player').agg(**{'Partidas Jogadas': ('match_id', 'nunique'),'Abates (K)': ('k', 'sum'),'Assistências (A)': ('a', 'sum'),'Mortes (D)': ('d', 'sum')})
        wins = self.df[self.df['won'] == True].groupby('player')['won'].count()
        losses = self.df[self.df['won'] == False].groupby('player')['won'].count()
        stats['Vitórias'] = wins.reindex(stats.index, fill_value=0)
        stats['Derrotas'] = losses.reindex(stats.index, fill_value=0)
        stats['% de Vitória'] = (stats['Vitórias'] / (stats['Vitórias'] + stats['Derrotas'])).fillna(0) * 100
        stats['Taxa K/D'] = (stats['Abates (K)'] / stats['Mortes (D)']).fillna(0)
        stats = stats[['Partidas Jogadas', 'Vitórias', 'Derrotas', '% de Vitória', 'Abates (K)', 'Assistências (A)', 'Mortes (D)', 'Taxa K/D']]
        return stats.sort_values(by=sort_by, ascending=False)

    def get_map_leaderboard(self, map_name, sort_by='% de Vitória'):
        map_df = self.df[self.df['map'] == map_name]
        if map_df.empty: return pd.DataFrame()
        stats = map_df.groupby('player').agg(**{'Partidas Jogadas': ('match_id', 'nunique'), 'Abates (K)': ('k', 'sum'), 'Mortes (D)': ('d', 'sum'), 'Saldo de Rounds': ('round_diff', 'sum')})
        wins = map_df[map_df['won'] == True].groupby('player')['won'].count()
        losses = map_df[map_df['won'] == False].groupby('player')['won'].count()
        stats['Vitórias'] = wins.reindex(stats.index, fill_value=0)
        stats['Derrotas'] = losses.reindex(stats.index, fill_value=0)
        stats['% de Vitória'] = (stats['Vitórias'] / (stats['Vitórias'] + stats['Derrotas'])).fillna(0) * 100
        stats['Taxa K/D'] = (stats['Abates (K)'] / stats['Mortes (D)']).fillna(0)
        return stats.sort_values(by=sort_by, ascending=False)
    
    def get_player_overall_stats_summary(self, player_name):
        player_df = self.df[self.df['player'] == player_name].dropna(subset=['won'])
        if player_df.empty: return None
        total_games = len(player_df)
        wins = player_df['won'].sum()
        win_rate = (wins / total_games) * 100 if total_games > 0 else 0
        return {"Partidas Jogadas": total_games, "Vitórias": int(wins), "Derrotas": total_games - int(wins), "% de Vitória": f"{win_rate:.2f}%"}

    def get_performance_by_map(self, player_name):
        player_df = self.df[self.df['player'] == player_name]
        if player_df.empty: return pd.DataFrame()
        stats_by_map = player_df.groupby('map').agg(Partidas=('match_id', 'nunique'), Abates=('k', 'sum'), Mortes=('d', 'sum'))
        wins_by_map = player_df[player_df['won'] == True].groupby('map')['won'].count()
        stats_by_map['Vitórias'] = wins_by_map.reindex(stats_by_map.index, fill_value=0)
        stats_by_map['% de Vitória'] = (stats_by_map['Vitórias'] / stats_by_map['Partidas']) * 100
        stats_by_map['K/D'] = (stats_by_map['Abates'] / stats_by_map['Mortes']).fillna(0)
        return stats_by_map.sort_values(by='Partidas', ascending=False)

    def get_performance_with_teammates(self, player_name, best=True):
        player_matches = self.df[self.df['player'] == player_name]
        if player_matches.empty: return pd.DataFrame()
        teammate_list = player_matches['teammates'].explode()
        common_teammates = teammate_list.value_counts()
        if common_teammates.empty: return pd.DataFrame()
        performance = []
        for teammate, games_played in common_teammates.items():
            matches_with_teammate = self.df[(self.df['player'] == player_name) & (self.df['teammates'].apply(lambda x: teammate in x))].dropna(subset=['won'])
            if not matches_with_teammate.empty:
                wins = matches_with_teammate['won'].sum()
                win_rate = (wins / len(matches_with_teammate)) * 100
                performance.append({'Companheiro': teammate, 'Partidas Juntos': games_played, '% de Vitória Juntos': win_rate})
        if not performance: return pd.DataFrame()
        return pd.DataFrame(performance).sort_values(by='% de Vitória Juntos', ascending=not best)

    def get_duo_stats(self, player1, player2):
        duo_matches = self.df[(self.df['player'] == player1) & (self.df['teammates'].apply(lambda x: player2 in x))].dropna(subset=['won'])
        if duo_matches.empty: return None
        total_games = len(duo_matches); wins = duo_matches['won'].sum(); win_rate = (wins / total_games) * 100
        p1_stats = duo_matches[['k', 'a', 'd']].sum()
        p2_matches = self.df[(self.df['player'] == player2) & (self.df['match_id'].isin(duo_matches['match_id']))]
        p2_stats = p2_matches[['k', 'a', 'd']].sum()
        combined_stats = p1_stats + p2_stats
        combined_kd = (combined_stats['k'] / combined_stats['d']) if combined_stats['d'] > 0 else 0
        return {"Partidas Juntos": total_games, "Vitórias": int(wins), "Derrotas": total_games - int(wins), "% de Vitória da Dupla": f"{win_rate:.2f}%", "K/D Combinado": f"{combined_kd:.2f}"}

    def get_h2h_overall(self, player1, player2):
        h2h_matches_p1 = self.df[(self.df['player'] == player1) & (self.df['opponents'].apply(lambda x: player2 in x))].dropna(subset=['won'])
        if h2h_matches_p1.empty: return None
        p1_wins = h2h_matches_p1['won'].sum(); total_games = len(h2h_matches_p1); p2_wins = total_games - p1_wins
        p1_total_k = h2h_matches_p1['k'].sum(); p1_total_d = h2h_matches_p1['d'].sum()
        p1_kd = p1_total_k / p1_total_d if p1_total_d > 0 else 0
        h2h_matches_p2 = self.df[(self.df['player'] == player2) & (self.df['match_id'].isin(h2h_matches_p1['match_id']))]
        p2_total_k = h2h_matches_p2['k'].sum(); p2_total_d = h2h_matches_p2['d'].sum()
        p2_kd = p2_total_k / p2_total_d if p2_total_d > 0 else 0
        return {"Partidas H2H": total_games, f"Vitórias {player1}": int(p1_wins), f"Vitórias {player2}": int(p2_wins), f"K/D Total {player1}": f"{p1_total_k}/{p1_total_d}", f"Taxa K/D {player1}": f"{p1_kd:.2f}", f"K/D Total {player2}": f"{p2_total_k}/{p2_total_d}", f"Taxa K/D {player2}": f"{p2_kd:.2f}"}

    def get_h2h_by_map(self, player1, player2):
        h2h_matches_p1 = self.df[(self.df['player'] == player1) & (self.df['opponents'].apply(lambda x: player2 in x))]
        if h2h_matches_p1.empty: return pd.DataFrame()
        results = []
        for map_name, group in h2h_matches_p1.groupby('map'):
            p1_stats = group.agg(K=('k', 'sum'), D=('d', 'sum'), RoundDiff=('round_diff', 'sum'))
            p1_wins = group['won'].sum()
            p2_matches_on_map = self.df[(self.df['player'] == player2) & (self.df['match_id'].isin(group['match_id']))]
            p2_stats = p2_matches_on_map.agg(K=('k', 'sum'), D=('d', 'sum'))
            results.append({'Mapa': map_name, 'Partidas': len(group), f'Vitórias {player1}': int(p1_wins), f'Vitórias {player2}': len(group) - int(p1_wins), f'K/D {player1}': f"{p1_stats.K}/{p1_stats.D}", f'K/D {player2}': f"{p2_stats.K}/{p2_stats.D}", 'Saldo de Rounds (p/ P1)': p1_stats.RoundDiff})
        return pd.DataFrame(results)

# --- INTERFACE GRÁFICA (Streamlit) ---

st.set_page_config(layout="wide", page_title="Análise de Partidas CS2")
st.title("📊 Painel de Análise de Partidas de Counter-Strike 2")

# A antiga função load_match_data_from_file() foi removida daqui.
# Agora, o upload é feito diretamente pelo componente do Streamlit.

uploaded_file = st.file_uploader("Carregue seu arquivo 'match_data.txt'", type="txt")

if uploaded_file is not None:
    try:
        string_data = uploaded_file.getvalue().decode("utf-8")
        match_data = [json.loads(line) for line in string_data.splitlines() if line.strip()]
        
        analyzer = StatsAnalyzer(match_data)
        player_list = analyzer.get_player_list()
        map_list = analyzer.get_map_list()

        st.sidebar.header("Selecione a Análise")
        analysis_type = st.sidebar.radio("Tipo de Análise:", ("Estatísticas Gerais", "Análise por Mapa", "Análise de Jogador", "Análise de Dupla (Juntos)", "Confronto 1x1 (Contra)"))

        # O restante da interface continua igual...
        if analysis_type == "Estatísticas Gerais":
            st.header("Estatísticas Gerais de Todos os Jogadores")
            sort_option = st.selectbox("Ordenar por:", ['Partidas Jogadas', '% de Vitória', 'Taxa K/D', 'Abates (K)'])
            stats_df = analyzer.get_overall_player_stats(sort_by=sort_option)
            st.dataframe(stats_df.style.format({'% de Vitória': '{:.2f}%', 'Taxa K/D': '{:.2f}'}))
        
        elif analysis_type == "Análise por Mapa":
            st.header("Ranking de Jogadores por Mapa")
            selected_map = st.selectbox("Selecione um Mapa:", map_list)
            if selected_map:
                st.subheader(f"Melhores jogadores em: {selected_map}")
                map_leaderboard = analyzer.get_map_leaderboard(selected_map)
                st.dataframe(map_leaderboard.style.format({'% de Vitória': '{:.2f}%', 'Taxa K/D': '{:.2f}'}))

        elif analysis_type == "Análise de Jogador":
            player_name = st.sidebar.selectbox("Selecione o Jogador:", player_list)
            if player_name:
                st.header(f"Análise Individual: {player_name}")
                overall_stats = analyzer.get_player_overall_stats_summary(player_name)
                if overall_stats:
                    c1, c2, c3, c4 = st.columns(4); c1.metric("Partidas", overall_stats["Partidas Jogadas"]); c2.metric("Vitórias", overall_stats["Vitórias"]); c3.metric("Derrotas", overall_stats["Derrotas"]); c4.metric("% de Vitória", overall_stats["% de Vitória"])
                st.subheader("Desempenho por Mapa"); st.dataframe(analyzer.get_performance_by_map(player_name).style.format({'% de Vitória': '{:.2f}%', 'K/D': '{:.2f}'}))
                c1, c2 = st.columns(2)
                with c1: st.subheader("Melhores Companheiros (% de Vitória)"); st.dataframe(analyzer.get_performance_with_teammates(player_name, best=True).head(5))
                with c2: st.subheader("Piores Companheiros (% de Vitória)"); st.dataframe(analyzer.get_performance_with_teammates(player_name, best=False).head(5))

        elif analysis_type == "Análise de Dupla (Juntos)":
            st.sidebar.subheader("Selecione a Dupla")
            player1 = st.sidebar.selectbox("Jogador 1:", player_list, index=player_list.index("Khan") if "Khan" in player_list else 0)
            player2 = st.sidebar.selectbox("Jogador 2:", player_list, index=player_list.index("Fzr") if "Fzr" in player_list else 1)
            if player1 and player2 and player1 != player2:
                st.header(f"Análise da Dupla: {player1} & {player2}"); duo_stats = analyzer.get_duo_stats(player1, player2)
                if duo_stats:
                    c1, c2, c3, c4, c5 = st.columns(5); c1.metric("Partidas Juntos", duo_stats["Partidas Juntos"]); c2.metric("Vitórias", duo_stats["Vitórias"]); c3.metric("Derrotas", duo_stats["Derrotas"]); c4.metric("% de Vitória", duo_stats["% de Vitória da Dupla"]); c5.metric("K/D Combinado", duo_stats["K/D Combinado"])
                else: st.warning("Esta dupla nunca jogou junta.")
        
        elif analysis_type == "Confronto 1x1 (Contra)":
            st.sidebar.subheader("Selecione os Jogadores")
            player1 = st.sidebar.selectbox("Jogador 1:", player_list, index=player_list.index("Khan") if "Khan" in player_list else 0)
            player2 = st.sidebar.selectbox("Jogador 2:", player_list, index=player_list.index("Ace") if "Ace" in player_list else 1)
            if player1 and player2 and player1 != player2:
                st.header(f"Confronto Direto: {player1} vs {player2}")
                h2h_overall = analyzer.get_h2h_overall(player1, player2)
                if h2h_overall:
                    st.subheader("Resumo Geral do Confronto")
                    c1, c2, c3 = st.columns(3); c1.metric("Partidas H2H", h2h_overall["Partidas H2H"]); c2.metric(f"Vitórias {player1}", h2h_overall[f"Vitórias {player1}"]); c3.metric(f"Vitórias {player2}", h2h_overall[f"Vitórias {player2}"])
                    c1, c2 = st.columns(2); c1.metric(f"Taxa K/D {player1}", h2h_overall[f"Taxa K/D {player1}"], help=h2h_overall[f"K/D Total {player1}"]); c2.metric(f"Taxa K/D {player2}", h2h_overall[f"Taxa K/D {player2}"], help=h2h_overall[f"K/D Total {player2}"])
                    st.subheader("Detalhes por Mapa"); h2h_map_stats = analyzer.get_h2h_by_map(player1, player2)
                    st.dataframe(h2h_map_stats)
                else: st.warning("Estes jogadores nunca se enfrentaram.")

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo. Verifique se o formato está correto. Detalhe: {e}")

else:
    st.info("Aguardando o upload do arquivo `match_data.txt` para iniciar a análise.")