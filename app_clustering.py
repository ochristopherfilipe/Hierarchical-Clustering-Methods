import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from io import BytesIO

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage


# Load the dataset
@st.cache_data
def uploaded_file(file):
    return pd.read_csv(file, infer_datetime_format=True, parse_dates=['Month'])


# Configuração da página
def main():
    st.set_page_config(
        page_title="Clusterização Hierárquica",
        layout="wide",
    )

    # Título e introdução
    st.title('Clusterização Hierárquica')
    st.write("""Vamos usar a base [online shoppers purchase intention](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) de Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018). [Web Link](https://doi.org/10.1007/s00521-018-3523-0).

A base trata de registros de 12.330 sessões de acesso a páginas, cada sessão sendo de um único usuário em um período de 12 meses, para posteriormente estudarmos a relação entre o design da página e o perfil do cliente - "Será que clientes com comportamento de navegação diferentes possuem propensão a compra diferente?" 

Nosso objetivo agora é agrupar as sessões de acesso ao portal considerando o comportamento de acesso e informações da data, como a proximidade a uma data especial, fim de semana e o mês.

        """)

    st.markdown("""   

                ### Essas são as informações do dataset:                        
        |Variavel                |Descrição          | 
        |------------------------|:-------------------| 
        |Administrative          | Quantidade de acessos em páginas administrativas| 
        |Administrative_Duration | Tempo de acesso em páginas administrativas | 
        |Informational           | Quantidade de acessos em páginas informativas  | 
        |Informational_Duration  | Tempo de acesso em páginas informativas  | 
        |ProductRelated          | Quantidade de acessos em páginas de produtos | 
        |ProductRelated_Duration | Tempo de acesso em páginas de produtos | 
        |BounceRates             | *Percentual de visitantes que entram no site e saem sem acionar outros *requests* durante a sessão  | 
        |ExitRates               | * Soma de vezes que a página é visualizada por último em uma sessão dividido pelo total de visualizações | 
        |PageValues              | * Representa o valor médio de uma página da Web que um usuário visitou antes de concluir uma transação de comércio eletrônico | 
        |SpecialDay              | Indica a proximidade a uma data festiva (dia das mães etc) | 
        |Month                   | Mês  | 
        |OperatingSystems        | Sistema operacional do visitante | 
        |Browser                 | Browser do visitante | 
        |Region                  | Região | 
        |TrafficType             | Tipo de tráfego                  | 
        |VisitorType             | Tipo de visitante: novo ou recorrente | 
        |Weekend                 | Indica final de semana | 
        |Revenue                 | Indica se houve compra ou não |
        """)

    st.markdown("---")
    st.markdown("---")
    st.markdown("---")
    # Configuração das abas
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Carregando O Arquivo", "Tratamento Os Dados", "Gráficos", "Modelo De Clustering", "Método Do Cotovelo"])

        
    with tab1:
        st.write('Carregando O Arquivo')
        file = st.file_uploader("Escolha um arquivo CSV", type=["csv"])

    with tab2:
        st.title('Tratamento Dos Dados')

        if file is not None:
            df = uploaded_file(file)
            if df.empty:
                st.warning("O arquivo está vazio.")
            else:
                # Display some information about the loaded DataFrame
                st.write("Número de linhas e colunas:", df.shape)
                st.write(df.head())  # Display the first few rows of the DataFrame

                # Verificar valores ausentes
                missing_values = df.isnull().sum()

                # Exibir as variáveis com valores ausentes, se houver
                st.subheader("Valores Ausentes:")
                st.write(missing_values[missing_values > 0])

                # Excluir linhas com valores ausentes
                df.dropna(inplace=True)

                # Confirmar que não há mais valores ausentes
                st.subheader("Após a exclusão de valores ausentes:")
                st.write(df.isnull().sum())

                # Lista de variáveis
                variaveis = ['Administrative', 'Administrative_Duration', 'Informational', 
                            'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 
                            'SpecialDay', 'Month', 'Weekend']
                variaveis_qtd = ['Administrative', 'Administrative_Duration', 'Informational', 
                                'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration']
                variaveis_cat = ['SpecialDay', 'Month', 'Weekend']
        
                st.markdown("---")
        
                st.subheader("Padronizando as colunas")
        
                # Criar DataFrame auxiliar
                df_pad = criar_df_auxiliar(df)

                # Exibir nomes das colunas
                st.write("Nomes das Colunas no DataFrame Transformado:")
                st.write(df_pad.columns.values)

    with tab3:
        st.title('Gráficos')
        if file is not None and not df.empty:
            # Visualizar a distribuição das variáveis numéricas
            numeric_columns = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                                'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']

            # Histogramas para variáveis numéricas
            df[numeric_columns].hist(bins=20, figsize=(15, 10))
            plt.suptitle('Distribuição das Variáveis Numéricas', y=1.02)
            st.pyplot(plt)

            # Contagem de valores para variáveis categóricas
            categorical_columns = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend', 'Revenue']

            for column in categorical_columns:
                plt.figure(figsize=(8, 5))
                sns.countplot(data=df, x=column)
                plt.title(f'Distribuição de {column}')
                st.pyplot(plt)
     
    with tab4:
        st.title('Modelo De Clustering')

        variaveis_qtd = ['Administrative', 'Administrative_Duration', 'Informational',
                        'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration']
        variaveis_cat = ['SpecialDay', 'Month', 'Weekend']

        @st.cache_data
        def criar_df_auxiliar(df):
            df_pad = pd.DataFrame()
            df_pad[variaveis_qtd] = df[variaveis_qtd]
            df_pad = pd.concat([df_pad, pd.get_dummies(df[variaveis_cat], drop_first=True)], axis=1)
            return df_pad
                
        # Criar DataFrame auxiliar
        df_pad = criar_df_auxiliar(df)

        scaler = StandardScaler()
        df_pad_scaled = scaler.fit_transform(df_pad)

        # Configura o modelo de clustering hierárquico aglomerativo com linkage "complete", sem limite de distância e 3 clusters
        clus = AgglomerativeClustering(linkage="complete", distance_threshold=None, n_clusters=3)

        # Ajusta o modelo aos dados padronizados
        clus.fit(df_pad_scaled)

        # Adiciona a coluna 'grupo' ao DataFrame original 'df'
        df['grupo'] = clus.labels_

        # Mostrar tabela de hierarquia cruzada
        st.subheader("Cruzando a tabela de hierarquia com a venda ou nao venda da tabela 'Revenue'")
        crosstab_result = pd.crosstab(df['Revenue'], df['grupo'])
        st.write(crosstab_result)

        st.markdown('---')

        st.subheader("Cria uma tabela de contingência explorando a relação entre os grupos, os sistemas operacionais e a variável de receita ('Revenue').")
        crosstab_result2 = pd.crosstab([df['OperatingSystems'], df['Revenue']], df['grupo'])
        st.write(crosstab_result2)

        with tab5:
            st.title('Método do Cotovelo para Determinar o Número Ótimo de Clusters')

            # Inicializa uma lista para armazenar as variabilidades intra-cluster para diferentes números de clusters
            inertia = []

            # Testa diferentes números de clusters (de 1 a 29) e calcula a variabilidade intra-cluster (inertia) para cada um
            for k in range(1, 30):
                # Configura o modelo KMeans com o número atual de clusters e uma semente aleatória para reprodutibilidade
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Defina n_init explicitamente
                # Ajusta o modelo aos dados padronizados
                kmeans.fit(df_pad_scaled)
                # Armazena a variabilidade intra-cluster na lista
                inertia.append(kmeans.inertia_)

            # Plota o gráfico do método do cotovelo
            st.pyplot(plt.figure(figsize=(10, 6)))
            plt.plot(range(1, 30), inertia, marker='o')
            plt.xlabel('Número de Clusters')
            plt.ylabel('Inércia (Variabilidade Intra-cluster)')
            plt.title('Método do Cotovelo para Determinar o Número Ótimo de Clusters')

            # Encontrar o índice do ponto de cotovelo
            elbow_index = 11  
            plt.axvline(x=elbow_index + 1, color='red', linestyle='--', label='Ponto de Cotovelo')  # +1 porque o índice começa em 0

            plt.legend()
            st.pyplot(plt)

            # Configura o modelo KMeans com 11 clusters e uma semente aleatória para reprodutibilidade
            kmeans = KMeans(n_clusters=11, random_state=42, n_init=10)

            # Ajusta o modelo aos dados padronizados
            kmeans.fit(df_pad_scaled)

            # Adiciona a coluna 'grupo' ao DataFrame original 'df'
            df['grupo'] = kmeans.labels_

            # Crosstab
            crosstab_result = pd.crosstab(df['Revenue'], df['grupo'])

            # Gráfico de barras agrupadas
            st.pyplot(plt.figure(figsize=(10, 6)))
            crosstab_result.plot(kind='bar', colormap='viridis')
            plt.title('Distribuição dos Grupos por Receita')
            plt.xlabel('Receita')
            plt.ylabel('Contagem')
            plt.legend(title='Grupo', bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(plt)
        
# Roda o aplicativo
if __name__ == "__main__":
    main()
