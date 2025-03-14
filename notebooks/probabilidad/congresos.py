#!/usr/bin/env python
# coding: utf-8

# ## Importamos las Librerías:

# In[58]:


import empiricaldist
import matplotlib.pyplot as plt
import janitor
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import sklearn.metrics
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats as ss
import session_info
import warnings
warnings.filterwarnings('ignore')


# In[59]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style(style='whitegrid')
sns.set_context(context='notebook')
plt.rcParams['figure.figsize'] = (11, 9.4)


# ## Cargamos el Data Frame:

# In[3]:


datos = pd.read_csv('~/trabajo_infantil/data/probabilidad/congresos_electos.csv')
datos


# In[4]:


datos['Cargo'].value_counts()


# ## Realizamos la Unión entre los Votos Obtenidos y las Personas Electas:

# ### Cargamos el Conjunto de Datos:

# In[5]:


votos = pd.read_csv('~/trabajo_infantil/data/probabilidad/congresos_votos.csv')
votos.head()


# ### Seleccionamos los Cargos de Diputación:

# In[6]:


datos.isnull().sum()


# In[7]:


datos = datos[datos['Cargo'] == 'Diputación']
datos


# In[8]:


datos['Formula'].value_counts()


# ## Conocemos las Columnas de los Conjuntos de Datos:

# In[9]:


print(datos.columns)
print(votos.columns)


# ## Hacemos el Merge:

# In[10]:


columnas_clave = ['Año', 'Nombre_estado', 'Municipio', 'Partido', 'Coalición']

for col in columnas_clave:
    datos[col] = datos[col].astype(str)
    votos[col] = votos[col].astype(str)


# In[11]:


union = pd.merge(
    datos,
    votos,
    on=['Año', 'Nombre_estado', 'Municipio', 'Partido', 'Coalición'],
    how='left'  # Cambia a 'left', 'right' o 'outer' según lo que necesites
)
union


# ## Esandarizamos los Tipos de Variable:

# In[12]:


union['Año'] = union['Año'].astype(int)
votos['Año'] = votos['Año'].astype(int)


# ## Conocemos las Columnas del conjunto de Datos:

# In[13]:


print(votos.columns)


# ## Observamos los Nombres dde las Columnas del Conjunto de Datos:

# In[14]:


print("Columnas en votos:", list(votos.columns))


# ## Guardamos en un nuevo objeto la Union:

# In[15]:


registros = union
registros


# ## Transformamos la Unión para Conocer, los Votos en caso de Coalición, y el Partido que más Contribuyó a las Coaliciones:

# In[16]:


# Revisamos las columnas de votos primero
print(votos.columns)

# 1️⃣ Seleccionar las columnas necesarias del dataframe de votos ganador
columnas_a_rellenar = ['Votos', 'Votos_validos', 
                       'Votos_candidato_sin_registro', 'Votos_nulos', 
                       'Total_de_votos', 'Lista_nominal']

# Nos aseguramos que estas columnas están
columnas_ganadores = ['Año', 'Nombre_estado', 'Municipio', 'Partido', 'Coalición'] + columnas_a_rellenar

# Filtramos y ordenamos los datos de votos, luego eliminamos duplicados por Año, Nombre_estado, Municipio y Coalición.
ganadores = (
    votos[columnas_ganadores]
    .sort_values(['Año', 'Nombre_estado', 'Municipio', 'Votos'], ascending=[True, True, True, False])  # Ordenar por los votos en orden descendente
    .drop_duplicates(subset=['Año', 'Nombre_estado', 'Municipio'])  # Eliminar duplicados, manteniendo el partido con más votos
    .rename(columns={'Partido': 'Partido_ganador'})  # Renombramos la columna para no tener conflictos
)

print(ganadores.columns)

# 2️⃣ Merge controlado
# Hacemos un merge con las columnas relevantes de 'ganadores' en 'union'
union = pd.merge(
    union, 
    ganadores, 
    on=['Año', 'Nombre_estado','Municipio'],  # Unimos por Año, Nombre_estado y Municipio, sin Coalición
    how='left', 
    suffixes=('', '_ganador')  # Usamos un sufijo para identificar las columnas de los ganadores
)

# 3️⃣ Rellenamos solo los NaN usando los valores de ganadores
for col in columnas_a_rellenar:
    # Si hay valores NaN en las columnas originales de 'union', los rellenamos con los valores correspondientes de 'ganadores'
    union[col] = union[col].fillna(union[f"{col}_ganador"])

# Rellenamos la columna 'Partido' con el valor de 'Partido_ganador' si hay NaN
union['Partido'] = union['Partido'].fillna(union['Partido_ganador'])


# ## Observamos el Data Frame:

# In[17]:


union


# ## Realizamos la Nueva Unión para Mitigar los Datos Sesgados:

# ### Transformamos los Datos:

# In[18]:


congresistas = votos
congresistas


# ## Creamos una Función que Nos Permite Conocer los Votos por Coalición y la Cantidad Aportada por Cada Integrante:

# In[19]:


# Función para sumar votos por municipio y actualizar los registros
def sumar_votos_por_municipio(df):
    # Crear una copia para no modificar el DataFrame original
    df = df.copy()

    # Agrupamos por las variables clave que definen un municipio
    grupos = df.groupby(['Año', 'ID_estado', 'Nombre_estado', 'Municipio'])

    # Crear una lista para guardar los nuevos registros procesados
    resultados = []

    for _, grupo in grupos:
        # Procesamos cada municipio de manera independiente
        partidos_compuestos = grupo[grupo['Partido'].str.contains("_", na=False)]

        # Actualizar votos de los partidos compuestos
        for _, row in partidos_compuestos.iterrows():
            componentes = row["Partido"].split("_")
            # Calcular la suma de votos de los componentes
            votos_componentes = grupo[grupo["Partido"].isin(componentes)]["Votos"].sum()
            # Sumar los votos al partido compuesto
            grupo.loc[grupo["Partido"] == row["Partido"], "Votos"] += votos_componentes

        # Guardar el grupo procesado
        resultados.append(grupo)

    # Combinar todos los grupos procesados en un solo DataFrame
    df_actualizado = pd.concat(resultados, ignore_index=True)
    return df_actualizado

# Aplicar la función al DataFrame "presidencia"
congresista_actualizada = sumar_votos_por_municipio(congresistas)

# Si quieres guardar el resultado
# presidencia_actualizada.to_csv("presidencia_actualizada.csv", index=False)

# Mostrar un ejemplo de los datos actualizados
print(congresista_actualizada.head())


# ## Conocemos las Columnas del Nuevo Conjunto de Datos:

# In[20]:


congresista_actualizada.columns


# ## Transformamos los Valores del Nuevo Conjunto de Datos:

# In[21]:


congresista_actualizada['Año'] = congresista_actualizada['Año'].astype(int)
datos['Año'] = datos['Año'].astype(int)


# ## Conocemos los Tipos de Variables del Conjunto de Datos:

# In[22]:


print(datos.dtypes)
print(congresista_actualizada.dtypes)


# ## Efecutuamos la Unión Preliminar:

# In[23]:


union_1 = pd.merge(
    datos,
    congresista_actualizada,
    on=['Año', 'Nombre_estado', 'Municipio', 'Partido', 'Coalición'],
    how='left'  # Cambia a 'left', 'right' o 'outer' según lo que necesites
)
union_1


# ## Corroboramos que los Datos de la Unión Preliminar Coincidan con la Final:

# In[24]:


print(union_1['Formula'].value_counts())
(union_1['Votos'].isnull().sum())


# In[60]:


(
    union
    .pipe(
        lambda df: sns.ecdfplot(
            data = df[df['Votos_ganador'] > 0],
            x = 'Votos_ganador',
            hue = 'Coalición',
            palette= 'viridis'
        )
    )
)

plt.title('Densidad de los Votos Conseguidos por las Candidaturas Electas', size = 16)
plt.xlabel('Votos')
plt.ylabel('Probabilidad')
plt.tight_layout()
plt.show()


# In[62]:


union.columns


# In[66]:


from plotnine import *
from plotnine import options
options.figure_size = (10, 6)

# Filtrar datos y crear el gráfico ECDF
grafico = (
    ggplot(union[union['Votos_ganador'] > 0], aes(x='Votos_ganador', color='Coalición')) +
    stat_ecdf() +
    labs(
        title="Densidad de los Votos Conseguidos por las Candidaturas Electas",
        x="Votos",
        y="Probabilidad"
    ) +
    facet_grid('Año~Sexo')+
    theme_minimal() +
    theme(
        plot_title=element_text(face="bold", ha="center", color="#34495E", size=12),
        axis_title_x=element_text(face="bold", color="#2C3E50", size=12),
        axis_title_y=element_text(face="bold", color="#2C3E50", size=12),
        axis_text=element_text(color="#34495E"),
        panel_grid_major=element_line(color="#D0D3D4", size=0.5),
        panel_grid_minor=element_blank()
    )
)

grafico += theme(plot_background=element_rect(fill="white"))
grafico.show()


# ## Realizamos la Unión Final:

# In[28]:


# Revisamos las columnas de votos primero
print(congresista_actualizada.columns)

# 1️⃣ Seleccionar las columnas necesarias del dataframe de votos ganador
columnas_a_rellenar = ['Votos', 'Votos_validos', 
                       'Votos_candidato_sin_registro', 'Votos_nulos', 
                       'Total_de_votos', 'Lista_nominal']

# Nos aseguramos que estas columnas están
columnas_ganadores = ['Año', 'Nombre_estado', 'Municipio', 'Partido', 'Coalición'] + columnas_a_rellenar

ganadores = (
    congresista_actualizada[columnas_ganadores]
    .sort_values(['Año', 'Nombre_estado', 'Municipio', 'Votos'], ascending=[True, True, True, False])  # Ordenar por los votos en orden descendente
    .drop_duplicates(subset=['Año', 'Nombre_estado', 'Municipio'])  # Eliminar duplicados, manteniendo el partido con más votos
    .rename(columns={'Partido': 'Partido_ganador'})  # Renombramos la columna para no tener conflictos
)

print(ganadores.columns)

# 2️⃣ Merge controlado
union_1 = pd.merge(
    union_1, 
    ganadores, 
    on=['Año', 'Nombre_estado','Municipio'],
    how='left', 
    suffixes=('', '_ganador')
)

# 3️⃣ Rellenamos solo los NaN usando los valores de ganadores
for col in columnas_a_rellenar:
    union_1[col] = union_1[col].fillna(union_1[f"{col}"])

# Rellenamos Partido
union_1['Partido'] = union_1['Partido'].fillna(union_1['Partido_ganador'])


# ## Corroboramos que los Datos de la Unión Final Coincidan:

# In[29]:


print(union_1['Formula'].value_counts())
(union_1['Votos'].isnull().sum())


# ## Observamos el Conjunto de Datos:

# In[30]:


union_1


# In[31]:


(
    union_1
    .pipe(
        lambda df: sns.ecdfplot(
            data = df[df['Votos_ganador'] > 0],
            x = 'Votos_ganador',
            hue = 'Coalición',
            palette= 'viridis'
        )
    )
)

plt.title('Densidad de los Votos Conseguidos por las Candidaturas Electas', size = 16)
plt.xlabel('Votos')
plt.ylabel('Probabilidad')
plt.tight_layout()
plt.show()


# ## Alternativa:

# In[32]:


# Crear la columna de votos ajustados inicialmente igual a 'Votos'
congresistas['Nuevo_Votos'] = congresistas['Votos']

# Iterar solo sobre los registros con '_'
for index, row in congresistas[congresistas['Partido'].str.contains("_", na=False)].iterrows():
    partidos = row['Partido'].split("_")  # Separar los partidos de la coalición

    # Filtrar los votos de los partidos individuales dentro del mismo municipio
    filtro = (
        (congresistas['Año'] == row['Año']) &
        (congresistas['ID_estado'] == row['ID_estado']) &
        (congresistas['Nombre_estado'] == row['Nombre_estado']) &
        (congresistas['Municipio'] == row['Municipio']) &
        (congresistas['Partido'].isin(partidos))
    )

    # Sumar los votos de los partidos involucrados en la coalición dentro del mismo municipio
    total_votos = congresistas.loc[filtro, 'Votos'].sum()

    # Asignar la suma a 'Nuevo_Votos' solo en la fila de la coalición
    congresistas.at[index, 'Nuevo_Votos'] = total_votos

# Mostrar los resultados
print(congresistas[['Año', 'ID_estado', 'Municipio', 'Partido', 'Votos', 'Nuevo_Votos']])


# ## Obervamos el Conjunto de Datos (Con este ya se Podría hacer un Nuevo Merge) como Unión_1:

# In[33]:


congresistas

