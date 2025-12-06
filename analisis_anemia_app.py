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
import datetime
from supabase import create_client, Client
import os

# Configuración de la página de Streamlit
st.set_page_config(page_title="Sistema de Salud Integral", layout="wide")

# ============================================
# CONFIGURACIÓN DE SUPABASE CON TUS CREDENCIALES
# ============================================

# TUS CREDENCIALES DE SUPABASE
SUPABASE_URL = "https://kwsuszkblbejvliniggd.supabase.co"
SUPABASE_KEY = "sb_publishable_DpWyb9LfXqiZBlmuSWfgIw_O2-LDm2b"

# Inicializar cliente de Supabase
@st.cache_resource
def init_supabase():
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Probar la conexión
        test_response = supabase_client.table('alertas_hemoglobina').select("*").limit(1).execute()
        st.sidebar.success("✅ Conectado a Supabase")
        return supabase_client
    except Exception as e:
        st.sidebar.error(f"❌ Error de conexión a Supabase: {str(e)[:100]}")
        return None

# Inicializar Supabase
supabase = init_supabase()

# ============================================
# NUEVA SECCIÓN: ESTADO NUTRICIONAL Y PROGRAMAS
# ============================================

def show_nutritional_status():
    """Muestra el estado nutricional y programas de apoyo social"""
    
    st.header("📋 Estado Nutricional y Programas de Apoyo Social")
    
    # Crear dos columnas para la sección de estado nutricional
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Estado Nutricional del Paciente")
        
        # Mostrar información del estado nutricional
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #2c6fbb;">
            <h3 style="color: #1a4d8c; margin-top: 0;">Estado Nutricional: NO EVALUABLE</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0;">
                <div>
                    <p style="margin: 0; color: #666; font-size: 0.9rem;">Peso para la Edad</p>
                    <p style="margin: 0; font-weight: 600; color: #f57c00;">Edad sin referencia</p>
                </div>
                <div>
                    <p style="margin: 0; color: #666; font-size: 0.9rem;">Talla para la Edad</p>
                    <p style="margin: 0; font-weight: 600; color: #f57c00;">Edad sin referencia</p>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; margin-top: 10px;">
                <span style="background-color: #e8f5e9; color: #2e7d32; padding: 5px 15px; border-radius: 20px; font-weight: 600;">Seguimiento activo: SÍ</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Gráfico de indicadores nutricionales
        st.subheader("Indicadores Antropométricos")
        
        # Crear datos para el gráfico
        indicators = ['Peso/Edad', 'Talla/Edad', 'Peso/Talla', 'IMC/Edad']
        values = [None, None, 20, None]
        colors = ['#ff9800', '#ff9800', '#2196f3', '#9e9e9e']
        
        # Crear gráfico de barras
        fig = go.Figure()
        
        for i, (indicator, value, color) in enumerate(zip(indicators, values, colors)):
            fig.add_trace(go.Bar(
                x=[indicator],
                y=[value if value is not None else 0],
                name=indicator,
                marker_color=color,
                text=[f'No evaluable' if value is None else f'{value}'],
                textposition='auto',
                hovertemplate=f"{indicator}<br>" + 
                            ("No evaluable" if value is None else f"Valor: {value}<br>") +
                            "<extra></extra>"
            ))
        
        fig.update_layout(
            title="Indicadores Nutricionales",
            yaxis_title="Percentil",
            yaxis_range=[0, 100],
            showlegend=False,
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Información del Paciente")
        
        # Mostrar fecha actual
        today = datetime.date.today()
        st.info(f"**Fecha:** {today.strftime('%d/%m/%Y')}")
        
        # Información adicional del paciente
        st.markdown("""
        <div style="background-color: #f0f7ff; padding: 15px; border-radius: 10px; margin-top: 10px;">
            <p style="margin: 0; font-weight: 600; color: #1a4d8c;">Paciente:</p>
            <p style="margin: 5px 0 15px 0; color: #333;">[Nombre del paciente]</p>
            
            <p style="margin: 0; font-weight: 600; color: #1a4d8c;">Edad:</p>
            <p style="margin: 5px 0 15px 0; color: #333;">[Edad]</p>
            
            <p style="margin: 0; font-weight: 600; color: #1a4d8c;">Última evaluación:</p>
            <p style="margin: 5px 0 0 0; color: #333;">[Fecha]</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Separador
    st.markdown("---")
    
    # Sección de programas de apoyo social
    st.subheader("🏥 Programas de Apoyo Social")
    st.write("Seleccione el programa social al que pertenece el beneficiario:")
    
    # Crear tres columnas para los programas
    prog_col1, prog_col2, prog_col3 = st.columns(3)
    
    with prog_col1:
        st.markdown("""
        <div style="background-color: #fff3e0; padding: 20px; border-radius: 10px; text-align: center; cursor: pointer; border: 2px solid transparent; transition: all 0.3s;">
            <div style="background: linear-gradient(135deg, #ff9800, #ff5722); width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 15px auto;">
                <span style="color: white; font-size: 24px;">🥛</span>
            </div>
            <h3 style="color: #e65100; margin-bottom: 10px;">Vaso de Leche</h3>
            <p style="color: #666; font-size: 0.9rem;">Apoyo alimentario para niños, gestantes y adultos mayores</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Seleccionar Vaso de Leche", key="vaso_leche_btn", use_container_width=True):
            st.session_state.selected_program = "Vaso de Leche"
            st.success("✅ Programa 'Vaso de Leche' seleccionado")
    
    with prog_col2:
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 20px; border-radius: 10px; text-align: center; cursor: pointer; border: 2px solid transparent; transition: all 0.3s;">
            <div style="background: linear-gradient(135deg, #2196f3, #0d47a1); width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 15px auto;">
                <span style="color: white; font-size: 24px;">🤝</span>
            </div>
            <h3 style="color: #1565c0; margin-bottom: 10px;">Programa Juntos</h3>
            <p style="color: #666; font-size: 0.9rem;">Transferencias condicionadas para familias en pobreza</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Seleccionar Juntos", key="juntos_btn", use_container_width=True):
            st.session_state.selected_program = "Programa Juntos"
            st.success("✅ Programa 'Juntos' seleccionado")
    
    with prog_col3:
        st.markdown("""
        <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; text-align: center; cursor: pointer; border: 2px solid transparent; transition: all 0.3s;">
            <div style="background: linear-gradient(135deg, #4caf50, #1b5e20); width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 15px auto;">
                <span style="color: white; font-size: 24px;">🍎</span>
            </div>
            <h3 style="color: #2e7d32; margin-bottom: 10px;">Qali Warma</h3>
            <p style="color: #666; font-size: 0.9rem;">Alimentación escolar para instituciones educativas públicas</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Seleccionar Qali Warma", key="qaliwarma_btn", use_container_width=True):
            st.session_state.selected_program = "Qali Warma"
            st.success("✅ Programa 'Qali Warma' seleccionado")
    
    # Mostrar programa seleccionado
    if 'selected_program' in st.session_state:
        st.markdown("---")
        st.subheader("📋 Programa Seleccionado")
        
        programs_info = {
            "Vaso de Leche": {
                "desc": "Proporciona apoyo alimentario a niños, madres gestantes y adultos mayores en situación de vulnerabilidad.",
                "beneficiarios": "Niños 0-13 años, gestantes, adultos mayores",
                "frecuencia": "Diaria"
            },
            "Programa Juntos": {
                "desc": "Transferencias condicionadas para familias en situación de pobreza y pobreza extrema.",
                "beneficiarios": "Familias en pobreza con niños/adolescentes",
                "frecuencia": "Bimestral"
            },
            "Qali Warma": {
                "desc": "Alimentación escolar para niños de instituciones educativas públicas.",
                "beneficiarios": "Estudiantes de inicial y primaria",
                "frecuencia": "Diaria (escolar)"
            }
        }
        
        selected = st.session_state.selected_program
        info = programs_info.get(selected, {})
        
        st.markdown(f"""
        <div style="background-color: #f0f7ff; padding: 20px; border-radius: 10px; border-left: 5px solid #2c6fbb;">
            <h3 style="color: #1a4d8c; margin-top: 0;">{selected}</h3>
            <p style="margin-bottom: 15px;"><strong>Descripción:</strong> {info.get('desc', '')}</p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div>
                    <p style="margin: 0; color: #666; font-size: 0.9rem;">Beneficiarios</p>
                    <p style="margin: 0; font-weight: 600;">{info.get('beneficiarios', '')}</p>
                </div>
                <div>
                    <p style="margin: 0; color: #666; font-size: 0.9rem;">Frecuencia</p>
                    <p style="margin: 0; font-weight: 600;">{info.get('frecuencia', '')}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📝 Continuar con el seguimiento", type="primary", use_container_width=True):
            st.info(f"🚀 Continuando con el seguimiento para el programa: **{selected}**")

# ============================================
# FUNCIÓN CORREGIDA: REGISTRO DE ALERTAS DE HEMOGLOBINA
# ============================================

def registrar_alerta_hemoglobina():
    """Función CORREGIDA para registrar alertas de hemoglobina en Supabase"""
    
    st.header("🔴 Registro de Alertas de Hemoglobina (CORREGIDO)")
    
    # Verificar conexión a Supabase
    if supabase is None:
        st.error("❌ No hay conexión a Supabase. Verificando credenciales...")
        
        # Mostrar credenciales (solo para debug)
        with st.expander("🔧 Verificar configuración"):
            st.write(f"URL: {SUPABASE_URL}")
            st.write(f"Key: {SUPABASE_KEY[:20]}...")
        
        return
    
    st.success("✅ Conectado a Supabase correctamente")
    
    # Formulario para ingresar datos
    with st.form("form_alerta_hemoglobina_corregido"):
        st.subheader("📝 Ingresar Datos del Paciente")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dni = st.text_input("DNI del paciente*", max_chars=8, help="8 dígitos")
            nombre_completo = st.text_input("Nombre completo del paciente*")
            hemoglobina = st.number_input("Nivel de Hemoglobina (g/dL)*", 
                                         min_value=5.0, max_value=20.0, 
                                         value=12.0, step=0.1,
                                         help="Valor entre 5.0 y 20.0 g/dL")
        
        with col2:
            edad = st.number_input("Edad (años)*", min_value=0, max_value=100, value=30)
            sexo = st.selectbox("Sexo*", ["Femenino", "Masculino"])
            region = st.selectbox("Región*", 
                                 ["LIMA", "AMAZONAS", "CUSCO", "AREQUIPA", 
                                  "LA LIBERTAD", "PIURA", "JUNIN", "OTRA"])
            peso = st.number_input("Peso (kg, opcional)", min_value=0.0, max_value=200.0, 
                                  value=None, placeholder="Opcional")
        
        # Determinar nivel de riesgo basado en hemoglobina
        if hemoglobina < 8:
            riesgo = "ALTO RIESGO (Alerta Clínica - ALTA)"
            sugerencias = "ACCION PRIORITARIA: Derivación inmediata a especialista y suplementación urgente"
        elif hemoglobina < 10:
            riesgo = "ALTO RIESGO (Alerta Clínica - MODERADA)"
            sugerencias = "Suplemento de hierro y control mensual. Evaluar causas secundarias"
        elif hemoglobina < 12:
            riesgo = "RIESGO MODERADO"
            sugerencias = "Dieta rica en hierro y evaluación médica en 2 semanas"
        elif hemoglobina < 13:
            riesgo = "BAJO RIESGO"
            sugerencias = "PREVENCIÓN: Mantener alimentación balanceada y control anual"
        else:
            riesgo = "NORMAL"
            sugerencias = "Valores normales. Mantener hábitos saludables"
        
        # Mostrar previsualización
        st.markdown("---")
        st.subheader("📊 Previsualización de la Alerta")
        
        col_pre1, col_pre2 = st.columns(2)
        with col_pre1:
            st.info(f"**Nivel de riesgo:** {riesgo}")
            st.info(f"**Sugerencias:** {sugerencias}")
        
        with col_pre2:
            st.info(f"**Región:** {region}")
            if peso:
                st.info(f"**Peso:** {peso} kg")
        
        # Información importante sobre columnas
        with st.expander("ℹ️ Información sobre columnas de la tabla"):
            st.warning("""
            **IMPORTANTE:** Usando las columnas CORRECTAS de tu tabla `alertas_hemoglobina`:
            
            ✅ **Columnas que SÍ existen:**
            - `DNI` (texto)
            - `nombre_apellide` (texto - tiene typo)
            - `riesgo` (texto)
            - `fecha_alerta` (fecha)
            - `sugerencias` (texto)
            - `regién` (texto - tiene typo)
            - `peso...` (numérico, opcional)
            
            ❌ **NO usar:** `interpretacion_hematologica` (no existe)
            """)
        
        # Botón para enviar
        submitted = st.form_submit_button("💾 Guardar Alerta en Supabase")
        
        if submitted:
            if not dni or not nombre_completo:
                st.warning("⚠️ Por favor, complete los campos obligatorios (*)")
                return
            
            try:
                # Fecha actual en formato correcto
                fecha_actual = datetime.datetime.now().strftime("%Y-%m-%d")
                
                # PREPARAR DATOS CON LAS COLUMNAS CORRECTAS
                datos_para_insertar = {
                    'DNI': dni,
                    'nombre_apellide': nombre_completo,  # COLUMNA CORRECTA (con typo)
                    'riesgo': riesgo,
                    'fecha_alerta': fecha_actual,
                    'sugerencias': sugerencias,
                    'regién': region  # COLUMNA CORRECTA (con typo)
                }
                
                # Agregar peso si está presente
                if peso is not None:
                    datos_para_insertar['peso...'] = float(peso)
                
                # Mostrar datos que se enviarán
                with st.expander("🔍 Ver datos a enviar"):
                    st.json(datos_para_insertar)
                    st.write("**Columnas usadas:**", list(datos_para_insertar.keys()))
                
                # Insertar en Supabase
                st.info("🔄 Insertando datos en Supabase...")
                
                response = supabase.table('alertas_hemoglobina').insert(datos_para_insertar).execute()
                
                if response.data:
                    st.success(f"✅ ¡Alerta guardada correctamente para {nombre_completo}!")
                    st.balloons()
                    
                    # Mostrar confirmación
                    col_success1, col_success2 = st.columns(2)
                    with col_success1:
                        st.metric("DNI", dni)
                        st.metric("Riesgo", riesgo)
                    
                    with col_success2:
                        st.metric("Hemoglobina", f"{hemoglobina} g/dL")
                        st.metric("Fecha", fecha_actual)
                    
                else:
                    st.error("❌ No se recibió respuesta de Supabase")
                    
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Error al guardar en Supabase: {error_msg}")
                
                # Análisis detallado del error
                with st.expander("🔧 Análisis detallado del error"):
                    st.write("**Error completo:**", error_msg)
                    
                    if "interpretacion_hematologica" in error_msg:
                        st.error("""
                        ❌ **ERROR DETECTADO:**
                        
                        Estás intentando usar la columna `interpretacion_hematologica` que NO existe en tu tabla.
                        
                        **SOLUCIÓN:**
                        1. Revisa TODO tu código
                        2. Busca donde dice `interpretacion_hematologica`
                        3. Cámbialo por una columna que SÍ exista
                        
                        **Tus columnas existentes son:**
                        - DNI
                        - nombre_apellide
                        - riesgo
                        - fecha_alerta
                        - sugerencias
                        - regién
                        - peso...
                        """)
                    
                    # Sugerencia para verificar la tabla
                    st.info("""
                    **Para verificar tu tabla en Supabase:**
                    1. Ve a https://kwsuszkblbejvliniggd.supabase.co
                    2. Inicia sesión
                    3. Ve a "Table Editor"
                    4. Selecciona la tabla `alertas_hemoglobina`
                    5. Verifica los nombres exactos de las columnas
                    """)

# ============================================
# FUNCIONES ORIGINALES DEL ANÁLISIS DE ANEMIAS
# ============================================

@st.cache_data
def load_data():
    # Cargar los datos
    try:
        data = pd.read_csv("diagnostico.csv")
    except FileNotFoundError:
        st.error("Error: 'diagnostico.csv' no encontrado. Asegúrate de que esté en la raíz de tu repositorio de GitHub.")
        st.stop()
    
    # Limpieza básica de datos
    data = data[data['NEUTp'] < 100]
    
    for col in ['HGB', 'RBC', 'HCT']:
        data = data[data[col] > 0]
    
    le = LabelEncoder()
    data['Diagnosis_encoded'] = le.fit_transform(data['Diagnosis'])
    
    return data, le

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
    st.write(f"**Número de características:** {len(data.columns) - 1}")
    
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
    
    selected_vars = st.multiselect(
        "Seleccione variables para visualizar:",
        data.columns[:-2],
        default=['HGB', 'RBC', 'MCV', 'MCH'],
        key='exploratory_vars_multiselect'
    )
    
    if selected_vars:
        st.write("### Distribución de Variables")
        cols = st.columns(2)
        for i, var in enumerate(selected_vars):
            with cols[i % 2]:
                fig = px.histogram(data, x=var, color='Diagnosis', nbins=30,
                                  title=f'Distribución de {var} por Diagnóstico',
                                  marginal="box")
                st.plotly_chart(fig, use_container_width=True)
        
        st.write("### Boxplots por Diagnóstico")
        selected_var_boxplot = st.selectbox("Seleccione variable para boxplot:", selected_vars, key='boxplot_var_selector')
        fig = px.box(data, x='Diagnosis', y=selected_var_boxplot, 
                     title=f'Distribución de {selected_var_boxplot} por Diagnóstico')
        st.plotly_chart(fig, use_container_width=True)
    
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
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    selected_var_stat = st.selectbox("Seleccione variable para análisis:", numeric_cols[:-1], key='stat_var_selector')
    
    st.write("### Estadísticas Descriptivas por Diagnóstico")
    if selected_var_stat:
        desc_stats = data.groupby('Diagnosis')[selected_var_stat].describe()
        st.write(desc_stats)

    st.write("### Violin Plots por Diagnóstico")
    if selected_var_stat:
        fig_violin = px.violin(data, x='Diagnosis', y=selected_var_stat, color='Diagnosis', box=True, 
                               labels={'x': 'Diagnóstico', 'y': f'Valor de {selected_var_stat}'},
                               title=f'Distribución de {selected_var_stat} por Tipo de Anemia (Violin Plot)')
        st.plotly_chart(fig_violin, use_container_width=True)

    st.write("### Análisis de Varianza (ANOVA)")
    groups = [data[data['Diagnosis'] == diagnosis][selected_var_stat] 
              for diagnosis in data['Diagnosis'].unique()]
    
    f_val, p_val = stats.f_oneway(*groups)
    st.write(f"**Valor F:** {f_val:.4f}")
    st.write(f"**Valor p:** {p_val:.4f}")
    
    if p_val < 0.05:
        st.success("Hay diferencias significativas entre los grupos (p < 0.05)")
        
        st.write("### Prueba Post-Hoc (Tukey HSD)")
        tukey = pairwise_tukeyhsd(endog=data[selected_var_stat], 
                                 groups=data['Diagnosis'],
                                 alpha=0.05)
        st.text(tukey.summary())
    else:
        st.warning("No hay diferencias significativas entre los grupos (p = 0.05)")
    
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
    
    features = data.select_dtypes(include=[np.number]).columns.drop(['Diagnosis_encoded'])
    X = data[features]
    y = data['Diagnosis_encoded']
    
    test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.4, 0.2, 0.05, key='test_size_slider')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.write(f"**Precisión del modelo:** {accuracy:.2f}")
    
    st.write("### Matriz de Confusión")
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm,
                   labels=dict(x="Predicho", y="Real", color="Cantidad"),
                   x=label_encoder.classes_,
                   y=label_encoder.classes_,
                   text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("### Reporte de Clasificación")
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    st.text(report)
    
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
    
    st.write("### Análisis de Componentes Principales (PCA)")
    
    features = data.select_dtypes(include=[np.number]).columns.drop(['Diagnosis_encoded'])
    X = data[features]
    y = data['Diagnosis']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=principal_components, 
                          columns=['PC1', 'PC2'])
    pca_df['Diagnosis'] = y
    
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Diagnosis',
                     title='PCA: Visualización 2D de los Datos')
    st.plotly_chart(fig, use_container_width=True)
    
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
    
    st.write("### Heatmap de Medias por Diagnóstico")
    mean_by_diagnosis = data.groupby('Diagnosis')[features].mean()
    fig = px.imshow(mean_by_diagnosis,
                   labels=dict(x="Característica", y="Diagnóstico", color="Valor Medio"),
                   x=features,
                   y=mean_by_diagnosis.index,
                   aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

# Función para recomendaciones
def recommendations():
    st.subheader("Recomendaciones Basadas en el Diagnóstico")
    st.write("Seleccione un tipo de anemia para ver las recomendaciones generales asociadas.")

    diagnosis_types = sorted(data['Diagnosis'].unique().tolist())
    st.write(f"**Diagnósticos disponibles:** {diagnosis_types}")
    
    selected_diagnosis = st.selectbox(
        "Seleccione un diagnóstico:",
        ['Seleccione uno...'] + diagnosis_types, 
        key='recommendation_diagnosis_selector'
    )

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
        - **Control de la enfermedad:** La prioridad es el manejo y tratamiento de la enfermedad crónica subyacente.
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
        - **Urgencia Médica y Especialista:** El diagnóstico de leucemia requiere atención médica URGENTE y especializada por un hematólogo oncólogo.
        - **Confirmación Diagnóstica:** Se requerirán pruebas adicionales (biopsia de médula ósea, análisis genéticos).
        - **Plan de Tratamiento:** El tratamiento variará enormemente según el tipo de leucemia.
        - **Manejo de Complicaciones:** Es fundamental el manejo de las complicaciones (infecciones, hemorragias, anemia severa).
        - **Apoyo Psicológico:** Un diagnóstico de leucemia es devastador. El apoyo psicológico para el paciente y la familia es crucial.
        - **Seguimiento Continuo:** Requiere seguimiento médico constante y de por vida.
        **¡Advertencia Importante!** Esta aplicación NO es un sustituto del consejo médico profesional.
        """
    }

    if selected_diagnosis != 'Seleccione uno...':
        if selected_diagnosis in all_recommendations:
            st.markdown(all_recommendations[selected_diagnosis])
        else:
            st.warning(f"No hay recomendaciones específicas disponibles para '{selected_diagnosis}' en este momento. Por favor, consulta a un profesional de la salud.")
    else:
        st.info("Por favor, selecciona un tipo de diagnóstico del menú desplegable para ver las recomendaciones.")

# ============================================
# NUEVA FUNCIÓN: VER ALERTAS REGISTRADAS
# ============================================

def ver_alertas_registradas():
    """Función para ver las alertas registradas en Supabase"""
    
    st.header("📋 Alertas de Hemoglobina Registradas")
    
    if supabase is None:
        st.error("❌ No hay conexión a Supabase.")
        return
    
    try:
        # Obtener alertas de Supabase
        st.info("🔄 Obteniendo datos de Supabase...")
        response = supabase.table('alertas_hemoglobina').select("*").order('fecha_alerta', desc=True).execute()
        
        if response.data:
            df_alertas = pd.DataFrame(response.data)
            
            st.success(f"✅ Se encontraron {len(df_alertas)} alertas registradas")
            
            # Mostrar tabla con opciones
            st.subheader("📊 Tabla de Alertas")
            
            # Filtrar por región si hay datos
            if 'regién' in df_alertas.columns and not df_alertas.empty:
                regiones = sorted(df_alertas['regién'].unique())
                region_seleccionada = st.selectbox("Filtrar por región:", ["Todas"] + list(regiones))
                
                if region_seleccionada != "Todas":
                    df_alertas = df_alertas[df_alertas['regién'] == region_seleccionada]
                    st.info(f"Mostrando {len(df_alertas)} alertas de {region_seleccionada}")
            
            # Mostrar tabla
            st.dataframe(df_alertas, use_container_width=True, hide_index=True)
            
            # Opción para descargar
            csv = df_alertas.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar datos como CSV",
                data=csv,
                file_name=f"alertas_hemoglobina_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
            
            # Estadísticas
            st.subheader("📈 Estadísticas Generales")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                alto_riesgo = len(df_alertas[df_alertas['riesgo'].str.contains('ALTO RIESGO', na=False)]) if 'riesgo' in df_alertas.columns else 0
                st.metric("Alertas de Alto Riesgo", alto_riesgo)
            
            with col2:
                if 'regién' in df_alertas.columns and not df_alertas['regién'].empty:
                    region_frecuente = df_alertas['regién'].mode().iloc[0] if not df_alertas['regién'].mode().empty else "N/A"
                    st.metric("Región más frecuente", region_frecuente)
                else:
                    st.metric("Región más frecuente", "N/A")
            
            with col3:
                if 'fecha_alerta' in df_alertas.columns and not df_alertas['fecha_alerta'].empty:
                    ultima_fecha = df_alertas['fecha_alerta'].max()
                    st.metric("Última alerta", ultima_fecha)
                else:
                    st.metric("Última alerta", "N/A")
            
            # Gráfico de distribución por riesgo
            if 'riesgo' in df_alertas.columns:
                st.subheader("📊 Distribución por Nivel de Riesgo")
                riesgo_counts = df_alertas['riesgo'].value_counts()
                
                if not riesgo_counts.empty:
                    fig = px.bar(risgo_counts, 
                                x=risgo_counts.index, 
                                y=risgo_counts.values,
                                title='Distribución de Alertas por Nivel de Riesgo',
                                labels={'x': 'Nivel de Riesgo', 'y': 'Cantidad'},
                                color=risgo_counts.values,
                                color_continuous_scale='RdYlGn_r')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Gráfico por región
            if 'regién' in df_alertas.columns:
                st.subheader("🗺️ Distribución por Región")
                region_counts = df_alertas['regién'].value_counts()
                
                if not region_counts.empty:
                    fig2 = px.pie(region_counts, 
                                 names=region_counts.index, 
                                 values=region_counts.values,
                                 title='Distribución de Alertas por Región',
                                 hole=0.3)
                    st.plotly_chart(fig2, use_container_width=True)
            
        else:
            st.info("📭 No hay alertas registradas en la base de datos. Usa la opción 'Registro de Alertas' para agregar la primera.")
            
    except Exception as e:
        st.error(f"❌ Error al obtener alertas: {str(e)}")
        
        # Información de ayuda
        with st.expander("🔧 Solucionar problemas"):
            st.write("""
            **Posibles soluciones:**
            1. Verifica que la tabla `alertas_hemoglobina` exista en Supabase
            2. Revisa los permisos de la tabla
            3. Verifica que estés usando la clave correcta
            4. Asegúrate de que la tabla tenga al menos una columna
            """)

# ============================================
# FUNCIÓN PARA VERIFICAR CONEXIÓN
# ============================================

def verificar_conexion_supabase():
    """Verifica la conexión a Supabase y muestra información"""
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔗 Estado de Conexión")
    
    if supabase:
        st.sidebar.success("✅ Conectado a Supabase")
        
        # Probar obtener datos
        try:
            response = supabase.table('alertas_hemoglobina').select("*").limit(1).execute()
            if response.data:
                st.sidebar.info(f"📊 Tabla 'alertas_hemoglobina': {len(response.data)} registros de prueba")
            else:
                st.sidebar.warning("📭 La tabla está vacía o no existe")
        except:
            st.sidebar.warning("⚠️ Error al acceder a la tabla")
    else:
        st.sidebar.error("❌ Sin conexión a Supabase")

# ============================================
# MAIN APP
# ============================================

# Título principal de la aplicación
st.title('🏥 Sistema de Salud Integral: Análisis de Anemias y Estado Nutricional')

# Cargar datos
data, label_encoder = load_data()

# Sidebar para navegación
st.sidebar.title("🔍 Navegación Principal")

# Actualizar las opciones del menú
app_mode = st.sidebar.selectbox(
    "Seleccione el módulo:",
    [
        "Estado Nutricional", 
        "Registro de Alertas Hemoglobina",  # NUEVO - CORREGIDO
        "Ver Alertas Registradas",          # NUEVO
        "Análisis de Anemias"
    ],
    key='app_mode_selector'
)

# Verificar conexión
verificar_conexion_supabase()

# Mostrar el módulo seleccionado
if app_mode == "Estado Nutricional":
    show_nutritional_status()

elif app_mode == "Registro de Alertas Hemoglobina":
    registrar_alerta_hemoglobina()

elif app_mode == "Ver Alertas Registradas":
    ver_alertas_registradas()

elif app_mode == "Análisis de Anemias":
    analysis_option = st.sidebar.selectbox(
        "Seleccione el tipo de análisis:",
        ["Exploración de Datos", "Análisis Estadístico", "Modelado Predictivo", "Visualización Avanzada", "Recomendaciones"],
        key='main_analysis_selector'
    )
    
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
st.sidebar.markdown("**📋 Notas:**")
st.sidebar.markdown("- Los datos han sido limpiados automáticamente para eliminar valores extremos")
st.sidebar.markdown("- Para análisis estadísticos, p < 0.05 se considera significativo")
st.sidebar.markdown("- Sistema desarrollado para uso del personal de salud")

# Información de configuración
with st.sidebar.expander("⚙️ Información de Conexión"):
    st.write(f"**URL Supabase:** {SUPABASE_URL}")
    st.write(f"**Clave:** {'*' * len(SUPABASE_KEY)}")
    st.write("**Tabla:** alertas_hemoglobina")
    st.write("**Columnas correctas:** DNI, nombre_apellide, riesgo, fecha_alerta, sugerencias, regién, peso...")

# Pie de página
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 20px;">
    <p>🏥 <strong>Sistema de Salud Integral</strong> - Ministerio de Salud</p>
    <p>✅ <strong>CONEXIÓN CORREGIDA:</strong> Usando columnas correctas de Supabase</p>
    <p>Esta información es confidencial y de uso exclusivo para el personal de salud autorizado.</p>
</div>
""", unsafe_allow_html=True)
