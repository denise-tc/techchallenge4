# Principais bibliotecas
# ==============================================================================

import streamlit as st
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from datetime import date
from datetime import timedelta

#Origem dos dados
# ==============================================================================
import yfinance as yf

with open('modelo_ltsm.pkl', 'rb') as file_2:
  modelo_ltsm = pkl.load(file_2)

# Plots
# ==============================================================================

plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 10

#Page config
def wide_space_default():
    st.set_page_config(layout='wide')

# ==============================================================================
wide_space_default()

#Gráfico padrão
def cria_grafico( df1, df2, df3, df4, titulo, fixa_data, len_fixa_data):
    if fixa_data ==1 :
      dt_ini = df1.index[-(len_fixa_data):]
    else:
      dt_ini = df1.index[:1]

    dt_fim = df1.index[-1:]

    layout = go.Layout(
                    title = titulo,
                    titlefont = dict(size=20))
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=df1.index.values, y=df1[df1.columns[0]].values,
                    mode='lines',
                    line_color = 'blue',
                    name=df1.columns[0]))
    
    if df2.shape[0] != 0 :
      fig.add_trace(go.Scatter(x=df2.index.values, y=df2[df2.columns[0]].values,
                    mode='lines',
                    line_color = 'red',
                    name=df2.columns[0]))
    
    if df3.shape[0] != 0 :
      fig.add_trace(go.Scatter(x=df3.index.values, y=df3[df3.columns[0]].values,
                    mode='lines',
                    line_color = 'green',
                    name=df3.columns[0]))
       
    if df4.shape[0] != 0 :
      fig.add_trace(go.Scatter(x=df4.index.values, y=df4[df4.columns[0]].values,
                    mode='lines',
                    line_color = 'orange',
                    name=df4.columns[0]))   

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
                        ])
            ),
        range=(dt_ini[0].strftime('%Y-%m-%d'), dt_fim[0].strftime('%Y-%m-%d'))
    )
    return fig


st.title('Tech Challenge - DTAT2 - Grupo 56 - Análise Petroleo Brent')
st.image('cabeçalho.jpg')
st.image('Refinaria.jpg')
st.divider()  
st.subheader('Grupo 56')
st.write(":point_right: Denise Oliveira     rm351364")
st.write(":point_right: Fabrício Carraro    rm350902")
st.write(":point_right: Luiz H. Spezzano    rm351120")
st.write(":point_right: Thayze Darnieri     rm349021")
st.divider()  
st.subheader('O Desafio')
st.write("Você foi contratado(a) para uma consultoria, e seu trabalho envolve analisar os dados de preço do petróleo brent, que pode ser encontrado no site do [ipea](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&amp;serid=1650971490&amp;oper=view). Essa base de dados histórica envolve duas colunas: data e preço (em dólares). Um grande cliente do segmento pediu para que a consultoria desenvolvesse um dashboard interativo e que gere insights relevantes para tomada de decisão. Além disso, solicitaram que fosse desenvolvido um modelo de Machine Learning para fazer o forecasting do preço do petróleo.")

st.write("Seu objetivo é:")
st.write(":black_medium_small_square: Criar um dashboard interativo com ferramentas à sua escolha.")
st.write(":black_medium_small_square: Seu dashboard deve fazer parte de um storytelling que traga insights relevantes sobre a variação do preço do petróleo, como situações geopolíticas, crises econômicas, demanda global por energia e etc. Isso pode te ajudar com seu modelo. É obrigatório que você traga pelo menos 4 insights neste desafio.")
st.write(":black_medium_small_square: Criar um modelo de Machine Learning que faça a previsão do preço do petróleo diariamente (lembre-se de time series). Esse modelo deve estar contemplado em seu storytelling e deve conter o código que você trabalhou, analisando as performances do modelo.")
st.write(":black_medium_small_square: Criar um plano para fazer o deploy em produção do modelo, com as ferramentas que são necessárias.")
st.write(":black_medium_small_square: Faça um MVP do seu modelo em produção utilizando o Streamlit.")

st.divider()  
st.subheader('Notas Iniciais do Grupo')
st.write("Ao invés de utilizarmos o site do IPEA, conforme sugestão do desafio, optamos por utilizar a carga da biblioteca do Yahoo Finance ( https://finance.yahoo.com/ ) pois elimina a necessidade de ter que importar planilha, tornando o código mais versátil")

st.divider()  
st.subheader('Vamos falar de Petróleo!!!')
st.write("É comemorado em 29 de setembro o Dia do Petróleo!! O petróleo é um dos motores da sociedade atual, motivo de guerras e um dos principais responsáveis pelas mudanças climáticas. Todos os dias, são extraídos no mundo mais de 80 milhões de barris de petróleo. Seu nome vem do latim e significa 'óleo de pedra'.")

st.write("O líquido viscoso conhecido como “ouro negro” é uma mistura de hidrocarbonetos — compostos que contêm na sua estrutura molecular, principalmente, carbono e hidrogênio. Ele é o resultado de um processo de transformação ocorrido ao longo de milhões de anos.")

st.write(" **Origem do petróleo** ")

st.write("Há inúmeras teorias sobre o surgimento do petróleo, porém, a mais aceita é que ele surgiu através de restos orgânicos de animais e vegetais depositados no fundo de lagos e mares sofrendo transformações químicas ao longo de milhares de anos. Substância inflamável possui estado físico oleoso e com densidade menor do que a água. Sua composição química é a combinação de moléculas de carbono e hidrogênio (hidrocarbonetos).")

st.write(" **Uso e derivados** ")

st.write("Além de gerar a gasolina, que serve de combustível para grande parte dos automóveis que circulam no mundo, vários produtos são derivados do petróleo como, por exemplo, a parafina, gás natural, GLP, produtos asfálticos, nafta petroquímica, querosene, solventes, óleos combustíveis, óleos lubrificantes, óleo diesel e combustível de aviação.")

st.write(" **Principais características químicas e físicas do petróleo** ")

st.write(":black_medium_small_square: O petróleo é tipicamente encontrado em estado líquido à temperatura ambiente. No entanto, sua consistência pode variar de um líquido fino e volátil a uma substância espessa e viscosa.")
st.write(":black_medium_small_square: A cor do petróleo bruto pode variar de um amarelo palha claro a um preto semelhante a alcatrão, dependendo de sua composição.")
st.write(":black_medium_small_square: O petróleo é geralmente menos denso que a água, o que permite que ele flutue em superfícies aquáticas. A densidade pode variar com base na composição específica do petróleo.")
st.write(":black_medium_small_square: A viscosidade do petróleo pode variar bastante. Os mais leves são menos viscosos e fluem facilmente, enquanto os mais pesados são mais viscosos e fluem mais lentamente.")
st.write(":black_medium_small_square: Do ponto de vista químico, o petróleo é composto principalmente de hidrocarbonetos (moléculas constituídas de átomos de hidrogênio e carbono). Ele também contém quantidades variáveis de outros compostos, incluindo compostos organossulfurados, oxigenados e materiais nitrogenados.")
st.write(":black_medium_small_square: O petróleo é uma mistura de muitos hidrocarbonetos diferentes, cada um com seu próprio ponto de ebulição. Essa propriedade é explorada no processo de refino, onde a destilação fracionada separa o óleo em diferentes componentes.")
st.write(":black_medium_small_square: Ele é hidrofóbico (não solúvel em água), mas é solúvel em solventes orgânicos. Essa propriedade leva à formação de manchas de óleo em corpos d'água.")
st.write(":black_medium_small_square: Ele é altamente inflamável, tornando-o uma valiosa fonte de combustível. Ele libera energia na forma de calor quando queimado.")
st.write(":black_medium_small_square: O petróleo bruto tem um odor distinto, muitas vezes forte, que pode variar com base em sua composição e estado de degradação.")
st.write(":black_medium_small_square: O petróleo pode sofrer várias reações químicas, como oxidação e polimerização. Ele também pode reagir com certos produtos químicos, levando à formação de novos compostos.")
st.write(":black_medium_small_square: Alguns componentes do petróleo bruto, especialmente certos hidrocarbonetos aromáticos policíclicos (PAHs), podem ser tóxicos para humanos e animais selvagens.")

st.write(" **Tipos de petróleo** ")

st.write(":black_medium_small_square: **Petróleo Brent**: petróleo produzido na região do Mar do Norte, provenientes dos sistemas de exploração petrolífera de Brent e Ninian. É o petróleo na sua forma bruta (crú) sem passar pelo sistema de refino.")
st.write(":black_medium_small_square: **Petróleo Light**: petróleo leve, sem impurezas, que já passou pelo sistema de refino.")
st.write(":black_medium_small_square: **Petróleo Naftênico**: petróleo com grande quantidade de hidrocarbonetos naftênicos.")
st.write(":black_medium_small_square: **Petróleo Parafínico**: petróleo com grande concentração de hidrocarbonetos parafínicos.")
st.write(":black_medium_small_square: **Petróleo Aromático**: com grande concentração de hidrocarbonetos aromáticos.")

st.write(" **Maiores Produtores Mundiais** ")

st.write(":black_medium_small_square: Os Estados Unidos da América são reconhecidos como o maior produtor de petróleo do planeta, com mais de 16 milhões de barris extraídos por dia.")
st.write(":black_medium_small_square: Logo em seguida vem a Arábia Saudita com uma produção média diária de aproximadamente 11 milhões de barris.")
st.write(":black_medium_small_square: Em terceiro lugar, temos a Rússia com uma produção média diária de 10 milhões de barris, aproximadamente.")
st.write(":black_medium_small_square: O Brasil ocupa a 9ª na produção mundial de petróleo")
st.write(":black_medium_small_square: Juntos, USA, Arábia Saudita e Russia são responsáveis por mais de 50% da produção diária de petroleo, quando somamos a quantidade total da produção dos Top 15 produtores mundiais e por 40% do volume total da produção diária.")
st.image('graf15Produtores.png')

st.write(" **Brent – O que é, significado e definição** ")

st.write(":black_medium_small_square: Brent é uma sigla, que normalmente acompanha a cotação do petróleo e indica a origem do óleo e o mercado onde ele é negociado. O petróleo Brent foi batizado assim porque era extraído de uma base da Shell chamada Brent.")
st.write(":black_medium_small_square: Atualmente, a palavra Brent designa todo o petróleo extraído no Mar do Norte e comercializado na Bolsa de Londres.")
st.write(":black_medium_small_square: A cotação do petróleo Brent é referência para os mercados europeu e asiático.")
st.write(":black_medium_small_square: O petróleo Brent é produzido próximo ao mar, então os custos de transporte são significativamente menores.")
st.write(":black_medium_small_square: O Brent tem uma qualidade menor, mas, se tornou um padrão do petróleo e tem maior preço por causa das exportações mais confiáveis.")

st.write(" **Cotação e principais influências no preço** ")

st.write("O petróleo nada mais é do que uma commodity negociada no mercado financeiro internacional, normalmente na forma de contratos no mercado futuro. Cada um desses contratos costuma ser composto por 100 barris e negociado em dólares. Por serem derivados do preço do barril e do câmbio da moeda norte-americana, esses contratos futuros são conhecidos como derivativos.")
st.write("É importante ter em mente que o preço do petróleo é sempre expresso por barril (bbl), sendo que um barril é equivalente, a aproximadamente, 159 litros.")
st.write("Como qualquer ativo financeiro cotado em Bolsa de Valores, a cotação do barril de petróleo é submetida a flutuações que dependem essencialmente da lei da oferta e da demanda.")
st.write("A oferta do mercado de petróleo é definida pela Organização dos Países Exportadores de Petróleo (Opep), que é encarregada de determinar quantos barris serão produzidos por dia. O objetivo da Opep é regular e estabilizar o mercado.")
st.write("Contudo, dois grandes produtores, Estados Unidos e Rússia, não fazem parte da Opep. Ou seja, a Opep não controla a totalidade da produção de petróleo bruto no mundo, apenas uma parte.")
st.write("Do lado da demanda, um aumento das necessidades em energia de um país que seja grande consumidor, pode ter uma influência relevante para a cotação do barril, elevando seu preço.")
st.write("Por outro lado, em momentos de crises e recessões, como foi o período da pandemia do coronavírus, o consumo do petróleo e seus derivados, como combustíveis e lubrificantes, tende a diminuir e por consequência, reduzir as cotações.")
st.write("Mudanças climáticas também podem afetar a cotação do petróleo pela ótica da demanda, por exemplo, um inverno rigoroso aumenta o consumo dos combustíveis (calefação), como resultado, tende a elevar o preço pago pelo barril de petróleo.")
st.write("Por fim, destacamos que outro fator que pode afetar o preço do petróleo são as tensões geopolíticas nos maiores países produtores de petróleo. Qualquer fator que afete a produção desses países poderá mudar a cotação do petróleo no mercado financeiro.")
st.write("Em suma, podemos concluir que a cotação do petróleo é afetada por fatores ambientais, econômicos, políticos e sociais.")

st.divider()  
st.write("Agora que nos contextualizamos um pouco sobre o petróleo, vamos aprofundar nossa análise na cotação do Petróleo Brent (Brent Crude Oil Last Day Financ (BZ=F) ) ao longo dos anos. Definimos como Início da nossa análise o ano de 2009, para termos um cenário dos últimos 15 anos.")


#Fonte dos índices e cotações:   https://finance.yahoo.com/
indice = "BZ=F"   # Brent Crude Oil Last Day Financ (BZ=F)
inicio = "2009-01-01" #Define a data de ínicio para importação dos dados
#Coleta dados históricos do índice de referência até a data corrente
dados_acao = yf.download(indice, inicio) #Quando a biblioteca é chamada sem uma data final, carrega as cotações até a data corrente
df_cotacoes = pd.DataFrame({indice: dados_acao['Close']})

st.write("Vamos verificar inicialmente o comportamento gráfico das cotações do índice no período:")

df1 = pd.DataFrame(df_cotacoes)
df1.rename(columns={indice: 'Brent Crude Oil'}, inplace = True)
df2 = pd.DataFrame(data=[])
df3 = pd.DataFrame(data=[])
df4 = pd.DataFrame(data=[])

fig = cria_grafico(df1, df2,df3,df4,'Histórico de Cotações Brent Crude Oil', 0,0 )
st.plotly_chart(fig, use_container_width=True)

st.write(" **Análises de Oscilações** ")
mean_value = df_cotacoes[indice].describe().loc[['mean']].mean()
media_valor_brent = round(float(mean_value), 2)
txt_analise_oscilacao = "No gráfico acima, podemos verificar algumas oscilações bruscas na cotação do Brent Crude Oil ao longo dos últimos 15 anos. Na média, as cotações ficaram na casa dos USD " + str(media_valor_brent) + ", sendo que em abril/2020 a cotação atingiu seu menor preço no período, ficando em USD 19,33 e em março/2022 atingiu o maior valor do período, ficando em USD 127,98. Abaixo vamos detalhar um pouco esta variações"
st.write(txt_analise_oscilacao)
pd.options.display.float_format = "{:.2f}".format
st.write(df_cotacoes[indice].describe().loc[['mean','min','max']])

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Crise na Libia", "Aumento da Demanda de Petróleo", "Sãnções Econômicas USA vs Irã","COVID-19","Embargo Econômico USA vs Rússia"])

with tab1:
    st.header(" **abril/2011** - Crise na Líbia")
    st.image("crise na Libia.jpg")
    st.write("A crise líbia refere-se à atual crise humanitária e instabilidade política que ocorre na Líbia iniciada com os protestos da Primavera Árabe, que resultou na Guerra Civil Líbia (2011), na intervenção militar estrangeira e na deposição e morte de Muammar Gaddafi; continuando com a instabilidade de segurança geral em todo o país e na eclosão de uma nova guerra civil em 2014. A atual crise na Líbia resultou até agora em dezenas de milhares de vítimas desde o início da violência no início de 2011. Durante as duas guerras civis, a produção da indústria de petróleo, economicamente fundamental para a Líbia, colapsou para uma pequena fracção do seu nível normal.")
    st.write("Na época, Várias companhias petrolíferas suspenderam a produção do petróleo na Líbia, entre elas a Total, da França, a Repsol, da Espanha, a austríaca OMV e a italiana ENI. A Wintershall, da Alemanha, também anunciou a suspensão de suas atividades no país norte-africano, que geravam cerca de 100 mil barris de petróleo por dia.")
    st.write("Em abril/2011 a Líbia produzia 1,6 milhões de barris de petróleo por dia e era responsável por 2% do petróleo extraído no mundo. Apenas no mercado europeu, o país era responsável por 10% do abastecimento. A Itália era sua maior compradora.")
    st.write("A atividade petrolífera é fundamental para a economia líbia, representando 95% de suas exportações e 30% de seu Produto Interno Bruto (PIB)")

with tab2:
    st.header(" **janeiro/2016** - Aumento da Demanda de Petróleo no Mercado Mundial")
    st.image("queda preço 2016.jpg")
    st.write("Desde maio/2015, o petróleo da Opep vinha registrando uma forte tendência de baixa, que o fez perder desde então quase US$ 40 por barril.")
    st.write("O baixo nível dos preços foi consequência de um excesso da oferta nos mercados, sobretudo nos Estados Unidos. A isso se acrescentou também a perspectiva de uma breve alta das exportações do produto da parte do Irã, após a retirada das sanções internacionais contra esse país no marco do acordo nuclear pactuado em julho/2015.")
    st.write("Na época, o barril de petróleo Brent ficou abaixo dos USD 30 pela primeira vez desde março de 2004, devido à inquietação dos investidores sobre a economia chinesa e também pelo excesso da oferta.")
    st.write("Os preços do petróleo acumulavam mais de um ano e meio de quedas, devido a um excesso de oferta dos mercados, que se agravou devido às preocupação com as turbulências financeiras na China na époco, segundo maior consumidor mundial da commodity.")

with tab3:
    st.header(" **agosto a outubro/2018** - Sanções Econômicas dos Estados Unidos Contra o Irã")
    st.image("sanção dos Estados Unidos Irã.jpg")
    st.write("Em agosto/2018 entrou em vigor uma série de sanções econômicas impostas pelos Estados Unidos ao Irã, que alegavam que regime de Teerã não cumpria os termos do acordo nuclear assinado em 2015.")
    st.write("Os USA já havia anunciado em maio/2018 a saída do acordo e o restabelecimento das sanções contra o Irã e as empresas internacionais que faziam negócios com o país.")
    st.write("Acertado em 2015 depois de dois anos de negociações entre o Irã, Estados Unidos, China, Reino Unido, França e Alemanha, este pacto permitiu remover uma na parte das sanções contra Teerã e conseguir o compromisso do regime islâmico iraniano de não ter bomba nuclear.")
    st.write("Segundo o Organismo Internacional de Energia Atômica (OIEA), o Irã respeitou as condições do acordo. Mas os EUA diziam que isso não era verdade, e o governo de Israel dizia ter documentos mostrando que o Irã seguia enriquecendo urânio.")
    st.write("Depois de sua saída unilateral, Washington indicou que as sanções seriam efetivadas de maneira imediata para os novos contratos e deu um prazo de 90 a 180 dias para que as multinacionais abandonassem suas atividades no Irã.")
    st.write("Em outubro/2018 os preços do petróleo tiveram um salto, avançando para níveis não vistos desde de novembro de 2014, devidoo ao temor do mercado de as exportações de óleo do terceiro maior produtor da Organização dos Países Exportadores de Petróleo (Opep) serem reduzidas.")

with tab4:
    st.header(" **abril/2020** - Paralisação da Economia Global devido à COVID-19")
    st.image("quedaPreçoCOVID19.jpg")
    st.write("Com a economia global paralisada devido à pandemia de coronavírus, a demanda por petróleo caiu e faltou espaço para armazenar os superestoques. Os preços do petróleo já vinham caindo drasticamente desde meados de março/2020 devido à crise do novo coronavírus. Também houve um desacordo entre o cartel da Organização dos Países Exportadores de Petróleo (Opep) e outros países produtores dessa matéria-prima. A gota d'água agora foi que certos contratos de entrega de petróleo estavam prestes a vencer, mas os armazéns estão cheios, pois há poucos compradores. E isso empurrou o preço do petróleo WTI (West Texas Intermediate) para um valor negativo na noite de segunda-feira 20/04/2020 – ou seja, os operadores estavam pagando para que outros investidores assumissem os contratos.")
    st.write(" O petróleo WTI tem a propriedade de ser armazenado de forma bastante unidimensional, porque os oleodutos terminam no estado de Oklahoma, onde se localiza a maior instalação de armazenamento de petróleo do mundo e onde as entregas são feitas. Na cidade de Cushing, no entanto, os reservatórios já estavam bem cheios. Os tanques que sobraram cobravam cerca de 10 dólares para armazenar cada barril. O petróleo Brent, produzido no Mar do Norte, na Europa, tem mais alternativas de entrega e por isso não sofreu uma baixa tão significativa quanto o WTI, porém atingiu o menor valor de cotação dos últimos 18 anos!!")

with tab5:
    st.header(" **março/2022** - Embargo Econômico dos USA contra a Russia")
    st.image("gerraUcrania.jpg")
    st.write("A Rússia é a segunda maior exportadora de petróleo no mundo, responsável por aproximadamente 11% da produção mundial. Porém, países como EUA e Reino Unido se negaram a importar combustível russo como forma de retaliar as ações do Kremlin na Ucrânia. Devido a isso, aproximadamente 7 milhões de barris deixaram de ser comercializados diariamente. Como defesa contra o embargo econômico, Putin decretou a proibição de importação e exportação de matérias-primas, o que afetou as atividades econômicas em diversos países, como o próprio EUA e a Alemanha.")
    st.write("Desde que o mercado notou a intenção da invasão russa no dia 21/02/2022, o preço do barril Brent já vinha valorizando. No dia 08/03/2022 O preço do barril de Brent, petróleo de referência na Europa, subiu mais de 5%, e atingiu o maior valor da série histórica, cotado a USD 127,98.")
    st.write("Na contra-mão deste movimento, os Emirados Árabes Unidos entraram com um pedido à Opep (Organização dos Países Exportadores de Petróleo) para elevar o nível de produção de petróleo dos países membros. O objetivo era aumentar a oferta e controlar os preços dos barris. A medida foi um aceno positivo ao mercado e no dia 10 de março de 2022, o barril já era cotado em USD 112,37, uma queda de 13% em relação ao dia anterior.")


st.divider()  
st.subheader('Modelo de Predição')
st.write("Foi realizado o comparativo entre cinco modelos diferentes: ARIMA, PROPHET, LSTM, LSTM Ajustado e XBooster")
st.write("O modelo LSTM foi o que apresentou os melhores resultados e, portanto, foi o escolhido para a predição")
st.write("O notebook com todos os modelos pode ser acessado em: ")

