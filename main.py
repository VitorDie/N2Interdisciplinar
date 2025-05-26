import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from collections import deque
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Configuração da página
st.set_page_config(
    page_title="Simulação M/M/c",
    page_icon="🏥",
    layout="wide"
)

class MMcSimulation:
    def __init__(self, inter_arrival_times, service_times, num_servers):
        self.inter_arrival_times = inter_arrival_times
        self.service_times = service_times
        self.num_servers = num_servers
        
        # Calcular taxas a partir dos dados
        self.arrival_rate = 1 / np.mean(inter_arrival_times)  # λ
        self.service_rate = 1 / np.mean(service_times)        # μ
        
        # Resultados da simulação
        self.results = {
            'customer_arrivals': [],
            'customer_wait_times': [],
            'customer_system_times': [],
            'queue_sizes_over_time': [],
            'time_points': [],
            'server_busy_times': [0] * num_servers
        }
    
    def run_simulation(self):
        """Executa a simulação discreta do sistema M/M/c"""
        
        # Criar eventos de chegada
        events = []
        arrival_time = 0
        
        for i, inter_time in enumerate(self.inter_arrival_times):
            arrival_time += inter_time
            events.append({
                'time': arrival_time,
                'type': 'arrival',
                'customer_id': i,
                'service_time': self.service_times[i] if i < len(self.service_times) else np.random.exponential(1/self.service_rate)
            })
        
        # Ordenar eventos por tempo
        events.sort(key=lambda x: x['time'])
        
        # Estado da simulação
        current_time = 0
        queue = deque()
        servers = [None] * self.num_servers  # None = livre, dict = ocupado
        
        # Listas para rastrear métricas
        customer_wait_times = []
        customer_system_times = []
        queue_sizes = []
        time_points = []
        server_start_times = [0] * self.num_servers
        
        for event in events:
            current_time = event['time']
            
            # Verificar servidores que terminaram o atendimento
            for i in range(self.num_servers):
                if servers[i] and servers[i]['end_time'] <= current_time:
                    # Servidor terminou - acumular tempo ocupado
                    self.results['server_busy_times'][i] += servers[i]['end_time'] - servers[i]['start_time']
                    servers[i] = None
                    
                    # Se há fila, atender próximo cliente
                    if queue:
                        next_customer = queue.popleft()
                        start_time = current_time
                        end_time = start_time + next_customer['service_time']
                        
                        servers[i] = {
                            'customer_id': next_customer['customer_id'],
                            'start_time': start_time,
                            'end_time': end_time
                        }
                        
                        # Calcular métricas do cliente que saiu da fila
                        wait_time = start_time - next_customer['arrival_time']
                        system_time = wait_time + next_customer['service_time']
                        
                        customer_wait_times.append(wait_time)
                        customer_system_times.append(system_time)
            
            # Processar chegada
            if event['type'] == 'arrival':
                # Procurar servidor livre
                server_found = False
                for i in range(self.num_servers):
                    if servers[i] is None:
                        # Atendimento imediato
                        start_time = current_time
                        end_time = start_time + event['service_time']
                        
                        servers[i] = {
                            'customer_id': event['customer_id'],
                            'start_time': start_time,
                            'end_time': end_time
                        }
                        
                        # Cliente não esperou
                        customer_wait_times.append(0)
                        customer_system_times.append(event['service_time'])
                        server_found = True
                        break
                
                if not server_found:
                    # Adicionar à fila
                    queue.append({
                        'customer_id': event['customer_id'],
                        'service_time': event['service_time'],
                        'arrival_time': current_time
                    })
            
            # Registrar estado atual
            queue_sizes.append(len(queue))
            time_points.append(current_time)
        
        # Salvar resultados
        self.results['customer_wait_times'] = customer_wait_times
        self.results['customer_system_times'] = customer_system_times
        self.results['queue_sizes_over_time'] = queue_sizes
        self.results['time_points'] = time_points
        self.results['total_simulation_time'] = current_time
        
        return self.calculate_theoretical_metrics()
    
    def calculate_theoretical_metrics(self):
        """Calcula métricas teóricas do modelo M/M/c"""
        λ = self.arrival_rate
        μ = self.service_rate
        c = self.num_servers
        ρ = λ / μ  # Intensidade de tráfego por servidor
        
        if λ >= c * μ:
            st.error(f"⚠️ Sistema instável! λ ({λ:.3f}) >= c*μ ({c*μ:.3f})")
            return None
        
        # P₀ - Probabilidade do sistema estar vazio
        sum_part = sum([(ρ**n) / math.factorial(n) for n in range(c)])
        p0_denominator = sum_part + ((ρ**c) / math.factorial(c)) * (c * μ) / (c * μ - λ)
        p0 = 1 / p0_denominator
        
        # P_espera - Probabilidade de esperar (fila não vazia)
        p_wait = (p0 * (ρ**c)) / (math.factorial(c)) * (c * μ) / (c * μ - λ)
        
        # Lq - Número médio na fila
        lq = (p0 * (ρ**(c+1))) / (math.factorial(c) * ((c * μ - λ)**2) / (c * μ))
        
        # Wq - Tempo médio de espera na fila
        wq = lq / λ
        
        # W - Tempo médio no sistema
        w = wq + (1/μ)
        
        # L - Número médio no sistema
        l = λ * w
        
        return {
            'P0': p0,
            'P_espera': p_wait,
            'Lq': lq,
            'Wq': wq,
            'W': w,
            'L': l,
            'lambda': λ,
            'mu': μ,
            'rho': ρ,
            'utilization': λ / (c * μ)
        }
    
    def get_server_utilization(self):
        """Calcula utilização real dos servidores da simulação"""
        total_time = self.results['total_simulation_time']
        utilizations = []
        
        for i in range(self.num_servers):
            util = self.results['server_busy_times'][i] / total_time if total_time > 0 else 0
            utilizations.append(util)
        
        return utilizations

def create_plots(simulation, theoretical_metrics):
    """Cria os 3 gráficos solicitados"""
    
    # 1. Tempo de espera por cliente
    fig1 = px.line(
        x=list(range(len(simulation.results['customer_wait_times']))),
        y=simulation.results['customer_wait_times'],
        title="Tempo de Espera por Cliente",
        labels={'x': 'Cliente #', 'y': 'Tempo de Espera (unidades)'},
        template='plotly_white'
    )
    fig1.add_hline(y=theoretical_metrics['Wq'], line_dash="dash", 
                   annotation_text=f"Wq teórico = {theoretical_metrics['Wq']:.3f}")
    
    # 2. Tamanho da fila ao longo do tempo
    fig2 = px.line(
        x=simulation.results['time_points'],
        y=simulation.results['queue_sizes_over_time'],
        title="Tamanho da Fila ao Longo do Tempo",
        labels={'x': 'Tempo', 'y': 'Tamanho da Fila'},
        template='plotly_white'
    )
    fig2.add_hline(y=theoretical_metrics['Lq'], line_dash="dash", 
                   annotation_text=f"Lq teórico = {theoretical_metrics['Lq']:.3f}")
    
    # 3. Tempo de ocupação dos servidores
    server_utils = simulation.get_server_utilization()
    fig3 = px.bar(
        x=[f'Servidor {i+1}' for i in range(len(server_utils))],
        y=[util * 100 for util in server_utils],
        title="Tempo de Ocupação dos Servidores (%)",
        labels={'x': 'Servidor', 'y': 'Ocupação (%)'},
        template='plotly_white'
    )
    fig3.add_hline(y=theoretical_metrics['utilization'] * 100, line_dash="dash", 
                   annotation_text=f"Utilização teórica = {theoretical_metrics['utilization']*100:.1f}%")
    
    return fig1, fig2, fig3

def generate_report_data(simulation, theoretical_metrics):
    """Gera dados para o relatório CSV"""
    
    # Métricas simuladas
    sim_p0 = len([w for w in simulation.results['customer_wait_times'] if w == 0]) / len(simulation.results['customer_wait_times'])
    sim_p_wait = len([w for w in simulation.results['customer_wait_times'] if w > 0]) / len(simulation.results['customer_wait_times'])
    sim_lq = np.mean(simulation.results['queue_sizes_over_time'])
    sim_wq = np.mean(simulation.results['customer_wait_times'])
    sim_w = np.mean(simulation.results['customer_system_times'])
    sim_l = sim_lq + simulation.num_servers * theoretical_metrics['utilization']
    
    report_data = {
        'Métrica': ['P₀', 'P_espera', 'Lq', 'Wq', 'W', 'L'],
        'Valor_Teórico': [
            theoretical_metrics['P0'],
            theoretical_metrics['P_espera'],
            theoretical_metrics['Lq'],
            theoretical_metrics['Wq'],
            theoretical_metrics['W'],
            theoretical_metrics['L']
        ],
        'Valor_Simulado': [sim_p0, sim_p_wait, sim_lq, sim_wq, sim_w, sim_l],
        'Diferença_%': [
            abs(theoretical_metrics['P0'] - sim_p0) / theoretical_metrics['P0'] * 100,
            abs(theoretical_metrics['P_espera'] - sim_p_wait) / theoretical_metrics['P_espera'] * 100 if theoretical_metrics['P_espera'] > 0 else 0,
            abs(theoretical_metrics['Lq'] - sim_lq) / theoretical_metrics['Lq'] * 100 if theoretical_metrics['Lq'] > 0 else 0,
            abs(theoretical_metrics['Wq'] - sim_wq) / theoretical_metrics['Wq'] * 100 if theoretical_metrics['Wq'] > 0 else 0,
            abs(theoretical_metrics['W'] - sim_w) / theoretical_metrics['W'] * 100,
            abs(theoretical_metrics['L'] - sim_l) / theoretical_metrics['L'] * 100
        ]
    }
    
    return pd.DataFrame(report_data)

def analyze_improvements(inter_arrival_times, service_times, current_servers, theoretical_metrics):
    """Analisa melhorias: +1 servidor e atendente mais rápido"""
    
    st.subheader("💡 Análises Solicitadas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Vale a pena adicionar mais um servidor?**")
        
        # Simular com +1 servidor
        sim_plus = MMcSimulation(inter_arrival_times, service_times, current_servers + 1)
        metrics_plus = sim_plus.run_simulation()
        
        if metrics_plus:
            reduction_wq = ((theoretical_metrics['Wq'] - metrics_plus['Wq']) / theoretical_metrics['Wq']) * 100
            reduction_lq = ((theoretical_metrics['Lq'] - metrics_plus['Lq']) / theoretical_metrics['Lq']) * 100
            
            st.write(f"📊 **Servidor atual:** {current_servers}")
            st.write(f"📊 **Com +1 servidor:** {current_servers + 1}")
            st.write(f"⏱️ **Redução Wq:** {reduction_wq:.1f}%")
            st.write(f"👥 **Redução Lq:** {reduction_lq:.1f}%")
            
            if reduction_wq > 25:
                st.success("✅ **RECOMENDAÇÃO: SIM!** Redução significativa nos tempos.")
            elif reduction_wq > 10:
                st.warning("⚠️ **RECOMENDAÇÃO: CONSIDERE.** Melhoria moderada.")
            else:
                st.error("❌ **RECOMENDAÇÃO: NÃO.** Melhoria muito pequena.")
    
    with col2:
        st.write("**Qual seria o impacto de um atendente mais rápido?**")
        
        # Simular com tempos de serviço 20% menores (atendente 25% mais rápido)
        faster_service_times = service_times * 0.8
        sim_faster = MMcSimulation(inter_arrival_times, faster_service_times, current_servers)
        metrics_faster = sim_faster.run_simulation()
        
        if metrics_faster:
            improvement_wq = ((theoretical_metrics['Wq'] - metrics_faster['Wq']) / theoretical_metrics['Wq']) * 100
            improvement_w = ((theoretical_metrics['W'] - metrics_faster['W']) / theoretical_metrics['W']) * 100
            
            st.write(f"🚀 **Atendente 25% mais rápido:**")
            st.write(f"⏱️ **Redução Wq:** {improvement_wq:.1f}%")
            st.write(f"🏥 **Redução W:** {improvement_w:.1f}%")
            
            if improvement_wq > 20:
                st.success("✅ **IMPACTO: ALTO!** Vale muito a pena investir em treinamento.")
            elif improvement_wq > 10:
                st.info("ℹ️ **IMPACTO: MODERADO.** Melhoria interessante.")
            else:
                st.warning("⚠️ **IMPACTO: BAIXO.** Melhoria pequena.")

# ===== INTERFACE STREAMLIT =====

st.title("🏥 Simulação de Filas M/M/c")
st.markdown("**Sistema de análise para filas com múltiplos servidores**")
st.markdown("---")

# Upload do arquivo CSV
st.subheader("📁 Carregar Dados")

uploaded_file = st.file_uploader(
    "Faça upload do arquivo CSV com dados de tempo:",
    type=['csv'],
    help="O arquivo deve conter colunas: 'tempo_entre_chegadas' e 'tempo_atendimento'"
)
if uploaded_file is not None:
    # Ler dados
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Arquivo carregado: {len(df)} registros")
        
        # Mostrar preview
        with st.expander("👀 Visualizar dados"):
            st.dataframe(df.head(10))
            st.write(f"**Colunas disponíveis:** {list(df.columns)}")
        
        # Selecionar colunas
        col1, col2 = st.columns(2)
        with col1:
            arrival_col = st.selectbox("Coluna tempo entre chegadas:", df.columns)
        with col2:
            service_col = st.selectbox("Coluna tempo de atendimento:", df.columns)
        
        # Extrair dados
        inter_arrival_times = df[arrival_col].dropna().values
        service_times = df[service_col].dropna().values
        
        # Configurações da simulação
        st.subheader("⚙️ Configurações")
        num_servers = st.slider("Número de servidores (c):", 1, 10, 2)
        
        if st.button("🚀 Executar Simulação", type="primary"):
            
            with st.spinner("Executando simulação..."):
                
                # Criar e executar simulação
                simulation = MMcSimulation(inter_arrival_times, service_times, num_servers)
                theoretical_metrics = simulation.run_simulation()
                
                if theoretical_metrics:
                    
                    # Exibir métricas principais
                    st.subheader("📊 Métricas do Sistema")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("P₀ (Sistema Vazio)", f"{theoretical_metrics['P0']:.4f}")
                        st.metric("P_espera (Prob. Espera)", f"{theoretical_metrics['P_espera']:.4f}")
                    
                    with col2:
                        st.metric("Lq (Média na Fila)", f"{theoretical_metrics['Lq']:.3f}")
                        st.metric("Wq (Tempo Médio Espera)", f"{theoretical_metrics['Wq']:.3f}")
                    
                    with col3:
                        st.metric("L (Média no Sistema)", f"{theoretical_metrics['L']:.3f}")
                        st.metric("W (Tempo Médio Sistema)", f"{theoretical_metrics['W']:.3f}")
                    
                    # Criar gráficos
                    st.subheader("📈 Visualizações")
                    fig1, fig2, fig3 = create_plots(simulation, theoretical_metrics)
                    
                    tab1, tab2, tab3 = st.tabs(["Tempo de Espera", "Tamanho da Fila", "Ocupação Servidores"])
                    
                    with tab1:
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with tab2:
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    with tab3:
                        st.plotly_chart(fig3, use_container_width=True)
                    
                    # Relatório detalhado
                    st.subheader("📋 Relatório Detalhado")
                    report_df = generate_report_data(simulation, theoretical_metrics)
                    st.dataframe(report_df, use_container_width=True)
                    
                    # Download CSV
                    csv_buffer = io.StringIO()
                    report_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download resultados.csv",
                        data=csv_buffer.getvalue(),
                        file_name="resultados.csv",
                        mime="text/csv"
                    )
                    
                    # Análises
                    analyze_improvements(inter_arrival_times, service_times, num_servers, theoretical_metrics)
    
    except Exception as e:
        st.error(f"❌ Erro ao processar arquivo: {str(e)}")

else:
    st.info("👆 Faça upload de um arquivo CSV para começar")
    
    # Exemplo de formato esperado
    st.subheader("📝 Formato do CSV")
    example_data = pd.DataFrame({
        'tempo_entre_chegadas': [2.1, 1.8, 3.2, 2.5, 1.9],
        'tempo_atendimento': [4.2, 3.8, 5.1, 4.7, 3.9]
    })
    st.dataframe(example_data)