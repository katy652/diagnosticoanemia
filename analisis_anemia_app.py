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
# CONFIGURACIÓN DE SUPABASE - CREDENCIALES ACTUALES
# ============================================

SUPABASE_URL = "https://kwsuszkblbejvliniggd.supabase.co"
SUPABASE_KEY = "sb_publishable_DpWyb9LfXqiZBlmuSWfgIw_O2-LDm2b"

# Inicializar cliente de Supabase
@st.cache_resource
def init_supabase():
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Test de conexión
        test_response = supabase_client.table('alertas_hemoglobina').select("dni").limit(1).execute()
        if test_response.data is not None:
            st.sidebar.success("✅ Conectado a Supabase")
            return supabase_client
        else:
            st.sidebar.error("❌ Tabla vacía o no existe")
            return None
    except Exception as e:
        st.sidebar.error(f"❌ Error de conexión: {str(e)[:100]}")
        return None

# Inicializar
supabase = init_supabase()

# ============================================
# FUNCIÓN PARA OBTENER NOMBRES REALES DE COLUMNAS
# ============================================

def obtener_columnas_reales():
    """Obtiene los nombres reales de las columnas de la tabla"""
    columnas_conocidas = {
        'dni': 'dni',
        'nombre': 'nombre_apellide',  # ¡CON TYPO!
        'hemoglobina': 'hemoglobina_dl1',  # Verificar si existe
        'riesgo': 'riesgo',
        'fecha': 'fecha_alerta',
        'sugerencias': 'sugerencias',
        'region': 'regién',  # ¡CON TYPO!
        'peso': 'peso_kg',  # Posible
        'altitud': 'altitud_msnm'
    }
    return columnas_conocidas

# ============================================
# FUNCIÓN CORREGIDA: REGISTRO EN SUPABASE
# ============================================

def registrar_alerta_corregida():
    """FUNCIÓN CORREGIDA - Usa nombres de columnas reales"""
    
    st.header("🔴 Registro de Alertas Hemoglobina (VERSIÓN CORREGIDA)")
    
    if supabase is None:
        st.error("No hay conexión a Supabase. Verifica tus credenciales.")
        return
    
    # Mostrar advertencia sobre nombres de columnas
    st.warning("""
    ⚠️ **ATENCIÓN:** Usando nombres REALES de columnas de tu tabla Supabase:
    
    ✅ **Columnas confirmadas que SÍ existen:**
    - `dni`
    - `nombre_apellide` (con typo)
    - `riesgo`
    - `fecha_alerta`
    - `sugerencias`
    - `regién` (con typo)
    
    ❓ **Columnas por verificar si existen:**
    - `hemoglobina_dl1`
    - `peso_kg`
    - `interpretacion_hematologica`
    """)
    
    # Formulario CORREGIDO
    with st.form("formulario_corregido"):
        col1, col2 = st.columns(2)
        
        with col1:
            dni = st.text_input("DNI*", max_chars=8)
            nombre = st.text_input("Nombre completo*")
            edad = st.number_input("Edad (meses)*", min_value=0, max_value=240, value=24)
        
        with col2:
            hemoglobina = st.number_input("Hemoglobina (g/dL)*", min_value=3.0, max_value=20.0, value=10.0, step=0.1)
            region = st.selectbox("Región*", ["LIMA", "AREQUIPA", "CUSCO", "PUNO", "JUNIN", "OTRA"])
            peso = st.number_input("Peso (kg, opcional)", min_value=0.0, value=None)
        
        # Sugerencias basadas en hemoglobina
        if hemoglobina < 8:
            riesgo = "ALTO RIESGO (Emergencia)"
            sugerencias = "Derivación inmediata a hospital"
        elif hemoglobina < 10:
            riesgo = "ALTO RIESGO (Urgente)"
            sugerencias = "Suplementación inmediata de hierro"
        elif hemoglobina < 12:
            riesgo = "RIESGO MODERADO"
            sugerencias = "Control médico en 2 semanas"
        else:
            riesgo = "BAJO RIESGO"
            sugerencias = "Seguimiento rutinario"
        
        submitted = st.form_submit_button("💾 Guardar en Supabase")
        
        if submitted:
            if not dni or not nombre:
                st.error("Complete DNI y nombre")
                return
            
            try:
                # Datos CON NOMBRES CORRECTOS
                datos = {
                    'dni': dni,
                    'nombre_apellide': nombre,  # NOMBRE CORRECTO CON TYPO
                    'riesgo': riesgo,
                    'fecha_alerta': datetime.date.today().isoformat(),
                    'sugerencias': sugerencias,
                    'regién': region  # NOMBRE CORRECTO CON TYPO
                }
                
                # Agregar campos opcionales SI EXISTEN
                if hemoglobina:
                    datos['hemoglobina_dl1'] = float(hemoglobina)  # POSIBLE COLUMNA
                
                if peso:
                    datos['peso_kg'] = float(peso)  # POSIBLE COLUMNA
                
                if edad:
                    datos['edad_meses'] = int(edad)  # POSIBLE COLUMNA
                
                # Mostrar datos que se enviarán
                st.info("📤 Enviando estos datos a Supabase:")
                st.json(datos)
                
                # Insertar en Supabase
                response = supabase.table('alertas_hemoglobina').insert(datos).execute()
                
                if response.data:
                    st.success(f"✅ ¡Alerta guardada exitosamente! ID: {response.data[0].get('dni', 'N/A')}")
                    st.balloons()
                    
                    # Mostrar resumen
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric("Paciente", nombre)
                        st.metric("Hemoglobina", f"{hemoglobina} g/dL")
                    with col_res2:
                        st.metric("Riesgo", riesgo)
                        st.metric("Región", region)
                else:
                    st.error("No se recibió respuesta de Supabase")
                    
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Error: {error_msg}")
                
                # Análisis del error
                if "interpretacion_hematologica" in error_msg:
                    st.error("""
                    **ERROR CRÍTICO:** Estás usando la columna `interpretacion_hematologica` que NO existe.
                    
                    **SOLUCIÓN:**
                    1. NO uses esa columna
                    2. Usa solo las columnas que SÍ existen
                    3. Las columnas seguras son: dni, nombre_apellide, riesgo, fecha_alerta, sugerencias, regién
                    """)
                
                if "hemoglobina_dl1" in error_msg:
                    st.warning("""
                    **ADVERTENCIA:** La columna `hemoglobina_dl1` puede no existir.
                    Intenta sin ese campo o verifica el nombre exacto.
                    """)

# ============================================
# FUNCIÓN PARA VER DATOS DE SUPABASE
# ============================================

def ver_datos_supabase():
    """Muestra los datos actuales de Supabase"""
    
    st.header("📊 Datos en Supabase")
    
    if supabase is None:
        st.error("Sin conexión")
        return
    
    try:
        # Obtener datos
        response = supabase.table('alertas_hemoglobina').select("*").execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            st.success(f"✅ {len(df)} registros encontrados")
            
            # Mostrar datos
            st.dataframe(df, use_container_width=True)
            
            # Mostrar columnas disponibles
            st.subheader("📋 Columnas disponibles en la tabla:")
            columnas_info = []
            for col in df.columns:
                tipo = str(df[col].dtype)
                no_nulos = df[col].notna().sum()
                ejemplos = df[col].dropna().head(3).tolist()
                columnas_info.append({
                    "Columna": col,
                    "Tipo": tipo,
                    "No Nulos": no_nulos,
                    "Ejemplos": str(ejemplos[:3])
                })
            
            st.table(pd.DataFrame(columnas_info))
            
            # Estadísticas
            st.subheader("📈 Estadísticas")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total registros", len(df))
            with col2:
                if 'riesgo' in df.columns:
                    alto_riesgo = df['riesgo'].str.contains('ALTO', na=False).sum()
                    st.metric("Alto riesgo", alto_riesgo)
            with col3:
                if 'regién' in df.columns:
                    regiones = df['regién'].nunique()
                    st.metric("Regiones", regiones)
            
        else:
            st.info("La tabla está vacía")
            
    except Exception as e:
        st.error(f"Error: {e}")

# ============================================
# FUNCIÓN PARA LIMPIAR Y REINICIAR TABLA
# ============================================

def reiniciar_tabla():
    """Opción para limpiar o reiniciar la tabla"""
    
    st.header("🔄 Gestión de Tabla Supabase")
    
    st.warning("""
    ⚠️ **ADVERTENCIA:** Esta operación puede eliminar datos.
    Solo para desarrollo/testing.
    """)
    
    # Opción 1: Ver estructura actual
    if st.button("🔍 Ver estructura actual de tabla"):
        try:
            # Obtener una fila para ver columnas
            response = supabase.table('alertas_hemoglobina').select("*").limit(1).execute()
            if response.data:
                fila = response.data[0]
                st.write("**Columnas actuales:**")
                for col in fila.keys():
                    st.write(f"- `{col}`: {type(fila[col]).__name__}")
            else:
                st.info("Tabla vacía o no existe")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Opción 2: Insertar datos de prueba
    st.subheader("Insertar datos de prueba")
    
    if st.button("➕ Insertar 3 pacientes de prueba"):
        try:
            datos_prueba = [
                {
                    'dni': '99999991',
                    'nombre_apellide': 'Paciente Prueba 1',
                    'riesgo': 'ALTO RIESGO',
                    'fecha_alerta': '2024-01-15',
                    'sugerencias': 'Control urgente',
                    'regién': 'LIMA',
                    'hemoglobina_dl1': 8.5
                },
                {
                    'dni': '99999992',
                    'nombre_apellide': 'Paciente Prueba 2',
                    'riesgo': 'MODERADO',
                    'fecha_alerta': '2024-01-16',
                    'sugerencias': 'Seguimiento mensual',
                    'regién': 'AREQUIPA',
                    'hemoglobina_dl1': 10.2
                },
                {
                    'dni': '99999993',
                    'nombre_apellide': 'Paciente Prueba 3',
                    'riesgo': 'BAJO RIESGO',
                    'fecha_alerta': '2024-01-17',
                    'sugerencias': 'Control anual',
                    'regién': 'CUSCO',
                    'hemoglobina_dl1': 12.5
                }
            ]
            
            response = supabase.table('alertas_hemoglobina').insert(datos_prueba).execute()
            if response.data:
                st.success("✅ 3 pacientes de prueba insertados")
                st.balloons()
        except Exception as e:
            st.error(f"Error: {e}")

# ============================================
# FUNCIONES ORIGINALES (sin cambios)
# ============================================

@st.cache_data
def load_data():
    try:
        data = pd.read_csv("diagnostico.csv")
    except FileNotFoundError:
        st.error("Error: 'diagnostico.csv' no encontrado.")
        st.stop()
    
    data = data[data['NEUTp'] < 100]
    for col in ['HGB', 'RBC', 'HCT']:
        data = data[data[col] > 0]
    
    le = LabelEncoder()
    data['Diagnosis_encoded'] = le.fit_transform(data['Diagnosis'])
    return data, le

def show_basic_info():
    st.subheader("Información Básica del Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Primeras filas:**")
        st.write(data.head())
    with col2:
        st.write("**Resumen estadístico:**")
        st.write(data.describe())

def exploratory_analysis():
    st.subheader("Análisis Exploratorio")
    selected_vars = st.multiselect("Seleccione variables:", data.columns[:-2], default=['HGB', 'RBC', 'MCV', 'MCH'])
    if selected_vars:
        cols = st.columns(2)
        for i, var in enumerate(selected_vars):
            with cols[i % 2]:
                fig = px.histogram(data, x=var, color='Diagnosis', nbins=30, title=f'Distribución de {var}')
                st.plotly_chart(fig, use_container_width=True)

def statistical_analysis():
    st.subheader("Análisis Estadístico")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    selected_var_stat = st.selectbox("Seleccione variable:", numeric_cols[:-1])
    if selected_var_stat:
        desc_stats = data.groupby('Diagnosis')[selected_var_stat].describe()
        st.write(desc_stats)

def predictive_modeling():
    st.subheader("Modelado Predictivo")
    features = data.select_dtypes(include=[np.number]).columns.drop(['Diagnosis_encoded'])
    X = data[features]
    y = data['Diagnosis_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Precisión:** {accuracy:.2f}")

def advanced_visualization():
    st.subheader("Visualización Avanzada")
    features = data.select_dtypes(include=[np.number]).columns.drop(['Diagnosis_encoded'])
    X = data[features]
    y = data['Diagnosis']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Diagnosis'] = y
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Diagnosis', title='PCA')
    st.plotly_chart(fig, use_container_width=True)

def recommendations():
    st.subheader("Recomendaciones")
    diagnosis_types = sorted(data['Diagnosis'].unique().tolist())
    selected_diagnosis = st.selectbox("Seleccione diagnóstico:", ['Seleccione...'] + diagnosis_types)
    if selected_diagnosis != 'Seleccione...':
        st.info(f"Recomendaciones para {selected_diagnosis}")

# ============================================
# MAIN APP - MENÚ SIMPLIFICADO
# ============================================

st.title('🏥 Sistema de Salud Integral - VERSIÓN CORREGIDA')

# Cargar datos
data, label_encoder = load_data()

# Sidebar con menú CORREGIDO
st.sidebar.title("🔍 Navegación")

app_mode = st.sidebar.selectbox(
    "Seleccione módulo:",
    [
        "📝 Registrar Alerta (CORREGIDO)",
        "📊 Ver Datos Supabase", 
        "🔄 Gestionar Tabla",
        "📈 Análisis de Anemias"
    ]
)

# Estado de conexión
st.sidebar.markdown("---")
if supabase:
    st.sidebar.success("✅ Conectado a Supabase")
    try:
        # Contar registros
        response = supabase.table('alertas_hemoglobina').select("dni", count='exact').execute()
        count = response.count if hasattr(response, 'count') else "?"
        st.sidebar.info(f"📊 {count} alertas registradas")
    except:
        st.sidebar.warning("⚠️ Error al contar registros")
else:
    st.sidebar.error("❌ Sin conexión")

# Ejecutar módulo seleccionado
if app_mode == "📝 Registrar Alerta (CORREGIDO)":
    registrar_alerta_corregida()
    
elif app_mode == "📊 Ver Datos Supabase":
    ver_datos_supabase()
    
elif app_mode == "🔄 Gestionar Tabla":
    reiniciar_tabla()
    
elif app_mode == "📈 Análisis de Anemias":
    analysis_option = st.sidebar.selectbox(
        "Tipo de análisis:",
        ["Exploración", "Estadística", "Modelado", "Visualización", "Recomendaciones"]
    )
    
    if analysis_option == "Exploración":
        show_basic_info()
        exploratory_analysis()
    elif analysis_option == "Estadística":
        statistical_analysis()
    elif analysis_option == "Modelado":
        predictive_modeling()
    elif analysis_option == "Visualización":
        advanced_visualization()
    elif analysis_option == "Recomendaciones":
        recommendations()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🏥 <strong>Sistema de Salud Integral</strong> - Versión Corregida</p>
    <p>✅ <strong>Usando nombres reales de columnas de Supabase</strong></p>
    <p>Para uso exclusivo del personal de salud autorizado</p>
</div>
""", unsafe_allow_html=True)
