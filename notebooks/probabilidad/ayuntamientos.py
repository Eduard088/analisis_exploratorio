#!/usr/bin/env python
# coding: utf-8

# ## Importamos las Librerías:

# In[3]:


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


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style(style='whitegrid')
sns.set_context(context='notebook')
plt.rcParams['figure.figsize'] = (11, 9.4)


# ## Cargamos el DataFrame:

# In[5]:


datos = pd.read_csv("~/trabajo_infantil/data/probabilidad/ayuntamientos_electos.csv")
datos


# In[6]:


datos['Cargo'].value_counts()


# ## Realizamos la Unión entre los Votos Obtenidos y las Personas Electas:

# ### Cargamos el Conjunto de Datos:

# In[7]:


votos = pd.read_csv('~/trabajo_infantil/data/probabilidad/ayuntamientos_votos.csv')
votos.head()


# ### Seleccionamos solo los Cargos de Ayuntamiento:

# In[8]:


datos = datos[datos['Cargo'] == 'Presidencia Municipal']
datos


# In[9]:


datos[datos['Municipio'] == 'MULEGE']


# In[10]:


votos[votos['Municipio'] == 'MULEGE']


# ### Conocemos las Columnas de Conjuntos de Datos:

# In[11]:


print(datos.columns)
print(votos.columns)


# In[12]:


print(datos.shape)
print(votos.shape)


# ## Hacemos el 'merge':

# In[13]:


columnas_clave = ['Año', 'Nombre_estado', 'ID_Municipio', 'Municipio', 'Partido']

for col in columnas_clave:
    datos[col] = datos[col].astype(str)
    votos[col] = votos[col].astype(str)


# In[14]:


union = pd.merge(
    datos,
    votos,
    on=['Año', 'Nombre_estado', 'ID_Municipio', 'Municipio', 'Partido'],
    how='left'  # Cambia a 'left', 'right' o 'outer' según lo que necesites
)
union


# #### Creamos una Copia de Unión, para Usarla Después:

# In[15]:


registros = union
registros


# In[16]:


union['Votos'].value_counts()


# ## Imputar los Valores Nan:

# In[17]:


# Revisamos las columnas de votos primero
print(votos.columns)

# 1️⃣ Seleccionar las columnas necesarias del dataframe de votos ganador
columnas_a_rellenar = ['Votos', 'Votos_validos', 
                       'Votos_candidato_sin_registro', 'Votos_nulos', 
                       'Total_de_votos', 'Lista_nominal']

# Nos aseguramos que estas columnas están
columnas_ganadores = ['Año', 'Nombre_estado', 'ID_Municipio', 'Municipio', 'Partido'] + columnas_a_rellenar

ganadores = (
    votos[columnas_ganadores]
    .sort_values(['Año', 'Nombre_estado', 'ID_Municipio', 'Municipio', 'Votos'], ascending=[True, True, True, True, False])
    .drop_duplicates(subset=['Año', 'Nombre_estado', 'ID_Municipio', 'Municipio'])
    .rename(columns={'Partido': 'Partido_ganador'})
)

print(ganadores.columns)

# 2️⃣ Merge controlado
union = pd.merge(
    union, 
    ganadores, 
    on=['Año', 'Nombre_estado', 'ID_Municipio', 'Municipio'],
    how='left', 
    suffixes=('', '_ganador')
)

# 3️⃣ Rellenamos solo los NaN usando los valores de ganadores
for col in columnas_a_rellenar:
    union[col] = union[col].fillna(union[f"{col}"])

# Rellenamos Partido
union['Partido'] = union['Partido'].fillna(union['Partido_ganador'])


# In[18]:


union


# In[20]:


union.columns


# ## Generamos el Conjunto de Datos para la Nueva Unión, con Menor Sesgo:

# ### Transormamos los datos:

# In[22]:


presidencias = votos
presidencias


# In[ ]:


# Función para sumar votos por municipio y actualizar los registros
def sumar_votos_por_municipio(df):
    # Crear una copia para no modificar el DataFrame original
    df = df.copy()

    # Agrupamos por las variables clave que definen un municipio
    grupos = df.groupby(['Año', 'ID_estado', 'Nombre_estado', 'ID_Municipio', 'Municipio'])

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
presidencia_actualizada = sumar_votos_por_municipio(presidencias)

# Si quieres guardar el resultado
# presidencia_actualizada.to_csv("presidencia_actualizada.csv", index=False)

# Mostrar un ejemplo de los datos actualizados
print(presidencia_actualizada.head())


# ## Conocemos las Columnas del Nuevo Conjunto de Datos:

# In[24]:


presidencia_actualizada.columns


# ## Transformamos los Valores del Nuevo Conjunto de Datos:

# In[25]:


presidencia_actualizada['Año'] = presidencia_actualizada['Año'].astype(int)
presidencia_actualizada['ID_Municipio'] = presidencia_actualizada['ID_Municipio'].astype(int)


datos['Año'] = datos['Año'].astype(int)
datos['ID_Municipio'] = datos['ID_Municipio'].astype(int)


# ## Conocemos los Tipos de Variables por Conjunto de Datos:

# In[26]:


print(datos.dtypes)
print(presidencia_actualizada.dtypes)


# ### Efecutuamos la Unión Preliminar:

# In[27]:


union_1 = pd.merge(
    datos,
    presidencia_actualizada,
    on=['Año', 'Nombre_estado', 'ID_Municipio', 'Municipio', 'Partido'],
    how='left'  # Cambia a 'left', 'right' o 'outer' según lo que necesites
)
union_1


# ### Realizamos la Unión Final:

# In[ ]:


# Revisamos las columnas de votos primero
print(presidencia_actualizada.columns)

# 1️⃣ Seleccionar las columnas necesarias del dataframe de votos ganador
columnas_a_rellenar = ['Votos', 'Votos_validos', 
                       'Votos_candidato_sin_registro', 'Votos_nulos', 
                       'Total_de_votos', 'Lista_nominal']

# Nos aseguramos que estas columnas están
columnas_ganadores = ['Año', 'Nombre_estado', 'ID_Municipio', 'Municipio', 'Partido'] + columnas_a_rellenar

ganadores = (
    presidencia_actualizada[columnas_ganadores]
    .sort_values(['Año', 'Nombre_estado', 'ID_Municipio', 'Municipio', 'Votos'], ascending=[True, True, True, True, False])
    .drop_duplicates(subset=['Año', 'Nombre_estado', 'ID_Municipio', 'Municipio'])
    .rename(columns={'Partido': 'Partido_ganador'})
)

print(ganadores.columns)

# 2️⃣ Merge controlado
union_1 = pd.merge(
    union_1, 
    ganadores, 
    on=['Año', 'Nombre_estado', 'ID_Municipio', 'Municipio'],
    how='left', 
    suffixes=('', '_ganador')
)

# 3️⃣ Rellenamos solo los NaN usando los valores de ganadores
for col in columnas_a_rellenar:
    union_1[col] = union_1[col].fillna(union_1[f"{col}"])

# Rellenamos Partido
union_1['Partido'] = union_1['Partido'].fillna(union_1['Partido_ganador'])


# In[34]:


union_1.columns


# In[29]:


union_1


# In[30]:


union['Partido_ganador'].isnull().sum()


# In[31]:


union_1['Partido_ganador'].isnull().sum()


# ## Graficamos los Resultados de la Primera Unión:

# In[ ]:


(
    union
    .pipe(
        lambda df: sns.ecdfplot(
            data = df[df['Votos_ganador'] > 0],
            x = 'Votos',
            hue = 'Sexo',
            palette= 'viridis'
        )
    )
)

plt.title('Probabilidad Acumulada de los Votos Conseguidos por las Candidaturas Electas', size = 16)
plt.xlabel('Votos')
plt.ylabel('Probabilidad')
plt.tight_layout()
plt.show()


# In[65]:


(
    union
    .pipe(
        lambda df: sns.kdeplot(
            data = df[df['Votos_ganador'] > 0],
            x = 'Votos_ganador',
            hue = 'Coalición_x',
            palette= 'viridis',
            bw_method= 0.3,
            fill = True
        )
    )
)

plt.title('Densidad de los Votos Conseguidos por las Candidaturas Electas', size = 16)
plt.xlabel('Votos')
plt.xlim(-35000, 100000)
plt.ylabel('Probabilidad')
plt.tight_layout()
plt.show()


# In[55]:


Votos_empirical = empiricaldist.Pmf.from_seq(
    union.Votos
)
Votos_empirical


# In[56]:


pmf_df = pd.DataFrame({'votos': Votos_empirical.index, 'probs': Votos_empirical.values})

pmf_df_sorted = pmf_df.sort_values(by='probs', ascending=False)

pmf_df_sorted


# ## Graficamos los Resultados de la Segunda Unión:

# In[72]:


(
    union_1
    .pipe(
        lambda df: sns.ecdfplot(
            data = df[df['Votos_ganador'] > 0],
            x = 'Votos_ganador',
            hue = 'Coalición_x',
            palette= 'viridis'
        )
    )
)

plt.title('Probabilidad Acumulada de los Votos Conseguidos por las Candidaturas Electas', size = 16)
plt.xlabel('Votos')
plt.ylabel('Probabilidad')
plt.tight_layout()
plt.show()


# In[80]:


(
    union_1
    .pipe(
        lambda df: sns.ecdfplot(
            data = df[df['Nombre_estado'] == 'YUCATAN'],
            x = 'Votos_ganador',
            hue = 'Sexo',
            palette= 'viridis'
        )
    )
)

plt.title('Probabilidad Acumulada de los Votos Conseguidos por las Candidaturas Electas', size = 16)
plt.xlabel('Votos')
plt.ylabel('Probabilidad')
plt.tight_layout()
plt.show()


# In[82]:


(
    union
    .pipe(
        lambda df: sns.kdeplot(
            data = df[df['Votos_ganador'] > 0],
            x = 'Votos_ganador',
            hue = 'Coalición_x',
            palette= 'viridis',
            bw_method= 0.3,
            fill = True
        )
    )
)

plt.title('Densidad de los Votos Conseguidos por las Candidaturas Electas', size = 16)
plt.xlabel('Votos')
plt.xlim(-35000, 100000)
plt.ylabel('Probabilidad')
plt.tight_layout()
plt.show()


# ## Seccionamos por Sexo los Conjuntos de Datos:

# ### Hombres:

# In[25]:


Hombres = datos[datos['Sexo'] == 'Hombre']
Hombres.head(5)


# ### Mujeres:

# In[26]:


Mujeres = datos[datos['Sexo'] == 'Mujer']
Mujeres.head(5)


# ## Transformamos los votos para añadirlo después a 'registros':

# In[32]:


presidencias = votos
presidencias


# ## Transformamos los Datos Para Reducir el Sesgo en los Datos, añadiendo una variable (ALTERNATIVA):

# In[35]:


# Crear la columna de votos ajustados inicialmente igual a 'Votos'
presidencias['Nuevo_Votos'] = presidencias['Votos']

# Iterar solo sobre los registros con '_'
for index, row in presidencias[presidencias['Partido'].str.contains("_", na=False)].iterrows():
    partidos = row['Partido'].split("_")  # Separar los partidos de la coalición

    # Filtrar los votos de los partidos individuales dentro del mismo municipio
    filtro = (
        (presidencias['Año'] == row['Año']) &
        (presidencias['ID_estado'] == row['ID_estado']) &
        (presidencias['Nombre_estado'] == row['Nombre_estado']) &
        (presidencias['ID_Municipio'] == row['ID_Municipio']) &
        (presidencias['Municipio'] == row['Municipio']) &
        (presidencias['Partido'].isin(partidos))
    )

    # Sumar los votos de los partidos involucrados en la coalición dentro del mismo municipio
    total_votos = presidencias.loc[filtro, 'Votos'].sum()

    # Asignar la suma a 'Nuevo_Votos' solo en la fila de la coalición
    presidencias.at[index, 'Nuevo_Votos'] = total_votos

# Mostrar los resultados
print(presidencias[['Año', 'ID_estado', 'ID_Municipio', 'Municipio', 'Partido', 'Votos', 'Nuevo_Votos']])


# In[36]:


presidencias['Nombre_estado'].value_counts()


# In[37]:


votos.shape


# In[38]:


presidencias.shape


# In[39]:


presidencias[(presidencias['Nombre_estado'] == 'BAJA CALIFORNIA SUR') & (presidencias['Año'] == "2018")]

