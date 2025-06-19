# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Configuración de la página de Streamlit
st.set_page_config(page_title="Análisis de Anemias", layout="wide")

@st.cache_data
def load_data():
    # Cargar los datos
    # ¡IMPORTANTE!: Asume que 'diagnostico.csv' está en la raíz de tu repositorio de GitHub
    try:
        data = pd.read_csv("diagnostico.csv")
    except FileNotFoundError:
        st.error("Error: 'diagnostico.csv' no encontrado. Asegúrate de que esté en la raíz de tu repositorio de GitHub.")
        st.stop() # Detiene la ejecución de la aplicación si el archivo no se encuentra
    
    # Limpieza básica de datos
    # Eliminar filas con valores extremadamente anómalos (como NEUTp=5317)
    data = data[data['NEUTp'] < 100]
    
    # Eliminar filas con valores negativos donde no deberían existir
    for col in ['HGB', 'RBC', 'HCT']:
        data = data[data[col] > 0]
    
    # Codificar la variable objetivo
    le = LabelEncoder()
    data['Diagnosis_encoded'] = le.fit_transform(data['Diagnosis'])
    
    return data, le

data, label_encoder = load_data()

# Título de la aplicación
st.title('Análisis de Datos de Clasificación de Tipos de Anemia')

# Sidebar para navegación
st.sidebar.title("Opciones de Análisis")
analysis_option = st.sidebar.selectbox(
    "Seleccione el tipo de análisis:",
    ["Exploración de Datos", "Análisis Estadístico", "Modelado Predictivo", "Visualización Avanzada", "Recomendaciones"],
    key='main_analysis_selector'
)

# Función para mostrar información básica del dataset
def show_basic_info():
    st.subheader("Información Básica del Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Primeras filas del dataset:**")
        st.write(data.head())
    
    with col2:
        st.write("**Resumen estadístico:**")
        st.write(data.describe())
    
    st.write(f"**Número total de muestras:** {len(data)}")
    st.write(f"**Número de características:** {len(data.columns) - 1}")  # Excluyendo Diagnosis
    
    st.write("**Distribución de diagnósticos:**")
    diagnosis_counts = data['Diagnosis'].value_counts()
    st.write(diagnosis_counts)
    
    fig = px.bar(diagnosis_counts, 
                 x=diagnosis_counts.index, 
                 y=diagnosis_counts.values,
                 labels={'x': 'Diagnóstico', 'y': 'Cantidad'},
                 title='Distribución de Tipos de Anemia')
    st.plotly_chart(fig, use_container_width=True)

# Función para análisis exploratorio
def exploratory_analysis():
    st.subheader("Análisis Exploratorio de Datos")
    
    # Selección de variables para visualización
    selected_vars = st.multiselect(
        "Seleccione variables para visualizar:",
        data.columns[:-2],  # Excluyendo Diagnosis y Diagnosis_encoded
        default=['HGB', 'RBC', 'MCV', 'MCH'],
        key='exploratory_vars_multiselect'
    )
    
    if selected_vars:
        # Histogramas
        st.write("### Distribución de Variables")
        cols = st.columns(2)
        for i, var in enumerate(selected_vars):
            with cols[i % 2]:
                fig = px.histogram(data, x=var, color='Diagnosis', nbins=30,
                                  title=f'Distribución de {var} por Diagnóstico',
                                  marginal="box")
                st.plotly_chart(fig, use_container_width=True)
        
        # Boxplots por diagnóstico
        st.write("### Boxplots por Diagnóstico")
        selected_var_boxplot = st.selectbox("Seleccione variable para boxplot:", selected_vars, key='boxplot_var_selector')
        fig = px.box(data, x='Diagnosis', y=selected_var_boxplot, 
                     title=f'Distribución de {selected_var_boxplot} por Diagnóstico')
        st.plotly_chart(fig, use_container_width=True)
    
    # Matriz de correlación
    st.write("### Matriz de Correlación")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0
    ))
    fig.update_layout(title='Matriz de Correlación')
    st.plotly_chart(fig, use_container_width=True)

# Función para análisis estadístico
def statistical_analysis():
    st.subheader("Análisis Estadístico")
    
    # Selección de variable para análisis
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    selected_var_stat = st.selectbox("Seleccione variable para análisis:", numeric_cols[:-1], key='stat_var_selector')
    
    # Estadísticas Descriptivas por Diagnóstico
    st.write("### Estadísticas Descriptivas por Diagnóstico")
    if selected_var_stat:
        desc_stats = data.groupby('Diagnosis')[selected_var_stat].describe()
        st.write(desc_stats)

    # Violin Plots
    st.write("### Violin Plots por Diagnóstico")
    if selected_var_stat:
        fig_violin = px.violin(data, x='Diagnosis', y=selected_var_stat, color='Diagnosis', box=True, 
                               labels={'x': 'Diagnóstico', 'y': f'Valor de {selected_var_stat}'},
                               title=f'Distribución de {selected_var_stat} por Tipo de Anemia (Violin Plot)')
        st.plotly_chart(fig_violin, use_container_width=True)

    # Análisis ANOVA
    st.write("### Análisis de Varianza (ANOVA)")
    groups = [data[data['Diagnosis'] == diagnosis][selected_var_stat] 
              for diagnosis in data['Diagnosis'].unique()]
    
    f_val, p_val = stats.f_oneway(*groups)
    st.write(f"**Valor F:** {f_val:.4f}")
    st.write(f"**Valor p:** {p_val:.4f}")
    
    if p_val < 0.05:
        st.success("Hay diferencias significativas entre los grupos (p < 0.05)")
        
        # Post-hoc test (Tukey HSD)
        st.write("### Prueba Post-Hoc (Tukey HSD)")
        tukey = pairwise_tukeyhsd(endog=data[selected_var_stat], 
                                 groups=data['Diagnosis'],
                                 alpha=0.05)
        st.text(tukey.summary())
    else:
        st.warning("No hay diferencias significativas entre los grupos (p = 0.05)")
    
    # Comparación de pares de diagnósticos con t-test
    st.write("### Comparación entre Pares de Diagnósticos (t-test)")
    diagnosis_pairs = st.multiselect(
        "Seleccione pares de diagnósticos para comparar:",
        [(a, b) for i, a in enumerate(data['Diagnosis'].unique()) 
         for b in list(data['Diagnosis'].unique())[i+1:]],
        format_func=lambda x: f"{x[0]} vs {x[1]}",
        key='ttest_pairs_multiselect'
    )
    
    for pair in diagnosis_pairs:
        group1 = data[data['Diagnosis'] == pair[0]][selected_var_stat] 
        group2 = data[data['Diagnosis'] == pair[1]][selected_var_stat] 
        
        t_val, p_val = stats.ttest_ind(group1, group2, equal_var=False)
        
        st.write(f"**{pair[0]} vs {pair[1]}**")
        st.write(f"T-valor: {t_val:.4f}, p-valor: {p_val:.4f}")
        st.write(f"Media {pair[0]}: {group1.mean():.2f}")
        st.write(f"Media {pair[1]}: {group2.mean():.2f}")
        st.write("---")

# Función para modelado predictivo
def predictive_modeling():
    st.subheader("Modelado Predictivo")
    
    # Selección de características y objetivo
    features = data.select_dtypes(include=[np.number]).columns.drop(['Diagnosis_encoded'])
    X = data[features]
    y = data['Diagnosis_encoded']
    
    # División de datos
    test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.4, 0.2, 0.05, key='test_size_slider')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Escalado de características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenamiento del modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluación del modelo
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.write(f"**Precisión del modelo:** {accuracy:.2f}")
    
    # Matriz de confusión
    st.write("### Matriz de Confusión")
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm,
                   labels=dict(x="Predicho", y="Real", color="Cantidad"),
                   x=label_encoder.classes_,
                   y=label_encoder.classes_,
                   text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Reporte de clasificación
    st.write("### Reporte de Clasificación")
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    st.text(report)
    
    # Importancia de características
    st.write("### Importancia de Características")
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(feature_importance, 
                 x='Importance', 
                 y='Feature',
                 orientation='h',
                 title='Importancia de Características')
    st.plotly_chart(fig, use_container_width=True)

# Función para visualización avanzada
def advanced_visualization():
    st.subheader("Visualización Avanzada")
    
    # PCA para reducción de dimensionalidad
    st.write("### Análisis de Componentes Principales (PCA)")
    
    # Selección de características
    features = data.select_dtypes(include=[np.number]).columns.drop(['Diagnosis_encoded'])
    X = data[features]
    y = data['Diagnosis']
    
    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=principal_components, 
                          columns=['PC1', 'PC2'])
    pca_df['Diagnosis'] = y
    
    # Visualización PCA
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Diagnosis',
                     title='PCA: Visualización 2D de los Datos')
    st.plotly_chart(fig, use_container_width=True)
    
    # Gráfico de pares
    st.write("### Gráfico de Pares (Pair Plot)")
    selected_vars = st.multiselect(
        "Seleccione hasta 5 variables para pair plot:",
        features,
        default=['HGB', 'RBC', 'MCV', 'MCH'],
        max_selections=5,
        key='pairplot_vars_multiselect'
    )
    
    if selected_vars:
        fig = px.scatter_matrix(data, 
                               dimensions=selected_vars,
                               color='Diagnosis',
                               title='Gráfico de Pares por Diagnóstico')
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap por diagnóstico
    st.write("### Heatmap de Medias por Diagnóstico")
    mean_by_diagnosis = data.groupby('Diagnosis')[features].mean()
    fig = px.imshow(mean_by_diagnosis,
                   labels=dict(x="Característica", y="Diagnóstico", color="Valor Medio"),
                   x=features,
                   y=mean_by_diagnosis.index,
                   aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

# FUNCIÓN: Recomendaciones
def recommendations():
    st.subheader("Recomendaciones Basadas en el Diagnóstico")
    st.write("Seleccione un tipo de anemia para ver las recomendaciones generales asociadas.")

    # Obtener los nombres de los diagnósticos únicos del dataset
    diagnosis_types = sorted(data['Diagnosis'].unique().tolist())
    st.write(f"**Diagnósticos disponibles:** {diagnosis_types}")
    
    selected_diagnosis = st.selectbox(
        "Seleccione un diagnóstico:",
        ['Seleccione uno...'] + diagnosis_types, 
        key='recommendation_diagnosis_selector'
    )

    # Diccionario de recomendaciones (¡ESTO DEBES PERSONALIZARLO CON INFORMACIÓN MÉDICA PRECISA!)
    all_recommendations = {
        'Anemia por deficiencia de hierro': """
        **Recomendaciones:**
        - **Consulta médica:** Es fundamental consultar a un médico para confirmar el diagnóstico y determinar la causa subyacente.
        - **Dieta:** Aumentar el consumo de alimentos ricos en hierro (carnes rojas, legumbres, espinacas, lentejas, cereales fortificados).
        - **Vitamina C:** Consumir alimentos ricos en Vitamina C (cítricos, brócoli) junto con las comidas ricas en hierro, ya que mejora su absorción.
        - **Suplementos:** Si es necesario, el médico podría recetar suplementos de hierro. No te automediques.
        - **Evitar inhibidores:** Limitar el consumo de té, café y calcio en las comidas ricas en hierro, ya que pueden inhibir su absorción.
        """,
        'Anemia por enfermedad crónica': """
        **Recomendaciones:**
        - **Control de la enfermedad:** La prioridad es el manejo y tratamiento de la enfermedad crónica subyacente (ej. enfermedad renal, inflamatoria, cáncer).
        - **Consulta médica:** Sigue las indicaciones de tu especialista.
        - **Nutrición:** Mantener una dieta equilibrada.
        - **Tratamientos específicos:** El médico podría considerar tratamientos como eritropoyetina o suplementos, según el caso.
        """,
        'Anemia aplásica': """
        **Recomendaciones:**
        - **Urgencia médica:** Requiere atención médica inmediata y seguimiento por un hematólogo.
        - **Evitar infecciones:** Es crucial prevenir infecciones debido al bajo recuento de glóbulos blancos.
        - **Tratamientos:** Puede incluir inmunosupresores, transfusiones de sangre o trasplante de médula ósea.
        - **Entorno:** Mantener un ambiente lo más estéril posible y evitar el contacto con personas enfermas.
        """,
        'Anemia megaloblástica': """
        **Recomendaciones:**
        - **Consulta médica:** Confirmar el diagnóstico (deficiencia de B12 o folato).
        - **Suplementos:** El médico prescribirá suplementos de vitamina B12 (inyecciones si la absorción es un problema) o ácido fólico.
        - **Dieta:** Incluir alimentos ricos en vitamina B12 (carnes, pescado, lácteos) y folato (vegetales de hoja verde, legumbres, cítricos).
        - **Causas subyacentes:** Investigar y tratar problemas de absorción o enfermedades que la causen.
        """,
        'Anemia hemolítica': """
        **Recomendaciones:**
        - **Consulta especializada:** Requiere evaluación por un hematólogo.
        - **Tratamiento de la causa:** El manejo dependerá de la causa subyacente (autoinmune, genética, medicamentos).
        - **Medicamentos:** Pueden incluir corticosteroides o inmunosupresores.
        - **Transfusiones:** Podrían ser necesarias en casos severos.
        - **Evitar desencadenantes:** Si es causada por medicamentos o exposiciones, identificarlas y evitarlas.
        """,
        'Normocytic normochromic anemia': """
        **Recomendaciones:**
        - **Consulta médica:** Este tipo de anemia puede tener muchas causas subyacentes (enfermedad crónica, pérdida aguda de sangre, enfermedad renal, etc.). Es esencial una evaluación médica para identificar la causa.
        - **Diagnóstico adicional:** Puede requerir pruebas adicionales para determinar la etiología, como pruebas de función renal o tiroidea, estudios de médula ósea o pruebas de inflamación.
        - **Tratamiento de la causa subyacente:** El tratamiento se centrará en la condición que está causando la anemia.
        - **Manejo de síntomas:** El médico puede sugerir tratamientos para aliviar síntomas como fatiga.
        """,
        'Leukemia': """
        **Recomendaciones (Leucemia):**
        - **Urgencia Médica y Especialista:** El diagnóstico de leucemia requiere atención médica URGENTE y especializada por un hematólogo oncólogo. No es algo que deba manejarse con recomendaciones generales de dieta o estilo de vida sin supervisión médica intensiva.
        - **Confirmación Diagnóstica:** Se requerirán pruebas adicionales (biopsia de médula ósea, análisis genéticos) para confirmar el tipo específico de leucemia y su estadio.
        - **Plan de Tratamiento:** El tratamiento variará enormemente según el tipo de leucemia (aguda/crónica, mieloide/linfoide), la edad del paciente y otros factores. Puede incluir quimioterapia, radioterapia, terapia dirigida, inmunoterapia o trasplante de células madre.
        - **Manejo de Complicaciones:** Es fundamental el manejo de las complicaciones (infecciones, hemorragias, anemia severa) que son comunes durante el tratamiento.
        - **Apoyo Psicológico:** Un diagnóstico de leucemia es devastador. El apoyo psicológico para el paciente y la familia es crucial.
        - **Seguimiento Continuo:** Requiere seguimiento médico constante y de por vida.
        **¡Advertencia Importante!** Esta aplicación NO es un sustituto del consejo médico profesional. Las recomendaciones para la leucemia son extremadamente complejas y deben ser proporcionadas ÚNICAMENTE por profesionales de la salud cualificados.
        """
           }

    if selected_diagnosis != 'Seleccione uno...':
        if selected_diagnosis in all_recommendations:
            st.markdown(all_recommendations[selected_diagnosis])
        else:
            st.warning(f"No hay recomendaciones específicas disponibles para '{selected_diagnosis}' en este momento. Por favor, consulta a un profesional de la salud.")
    else:
        st.info("Por favor, selecciona un tipo de diagnóstico del menú desplegable para ver las recomendaciones.")

# Mostrar el análisis seleccionado
if analysis_option == "Exploración de Datos":
    show_basic_info()
    exploratory_analysis()
elif analysis_option == "Análisis Estadístico":
    statistical_analysis()
elif analysis_option == "Modelado Predictivo":
    predictive_modeling()
elif analysis_option == "Visualización Avanzada":
    advanced_visualization()
elif analysis_option == "Recomendaciones":
    recommendations()

# Notas al pie
st.sidebar.markdown("---")
st.sidebar.markdown("**Notas:**")
st.sidebar.markdown("- Los datos han sido limpiados automáticamente para eliminar valores extremos")
st.sidebar.markdown("- Para análisis estadísticos, p < 0.05 se considera significativo")
