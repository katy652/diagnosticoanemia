# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import datetime
from supabase import create_client, Client

# Configuración de la página
st.set_page_config(page_title="Sistema de Hemoglobina", layout="wide")

# ============================================
# CONFIGURACIÓN SUPABASE - CREDENCIALES
# ============================================

SUPABASE_URL = "https://kwsuszkblbejvliniggd.supabase.co"
SUPABASE_KEY = "sb_publishable_DpWyb9LfXqiZBlmuSWfgIw_O2-LDm2b"

# Inicializar Supabase
@st.cache_resource
def init_supabase():
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Probar conexión
        test = supabase_client.table('alertas_hemoglobina').select("dni").limit(1).execute()
        if test.data is not None:
            st.sidebar.success("✅ Conectado a Supabase")
            return supabase_client
        else:
            st.sidebar.warning("⚠️ Tabla vacía")
            return supabase_client
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)[:100]}")
        return None

supabase = init_supabase()

# ============================================
# FUNCIÓN CORREGIDA - REGISTRO DE HEMOGLOBINA
# ============================================

def registrar_hemoglobina_corregido():
    """FUNCIÓN COMPLETAMENTE CORREGIDA - Sin columnas falsas"""
    
    st.title("🩸 Registro de Hemoglobina - VERSIÓN CORREGIDA")
    
    if supabase is None:
        st.error("❌ No hay conexión a Supabase")
        st.info(f"URL: {SUPABASE_URL}")
        st.info(f"Key: {SUPABASE_KEY[:20]}...")
        return
    
    # MOSTRAR COLUMNAS REALES QUE USAREMOS
    st.info("""
    ✅ **Usando SOLO columnas que SÍ existen:**
    - `dni` (texto, PRIMARY KEY)
    - `nombre_apellido` (texto)
    - `edad_meses` (entero)
    - `peso_kg` (número, opcional)
    - `talla_cm` (número, opcional)
    - `genero` (texto)
    - `region` (texto)
    - `hemoglobina_dl1` (número)
    - `interpretacion_hematologica` (texto)
    - `politicas_de_RLS` (texto)
    - `riesgo` (texto)
    - `estado_alerta` (texto)
    - `sugerencias` (texto)
    - `fecha_alerta` (fecha, automático)
    """)
    
    # FORMULARIO CORREGIDO
    with st.form("form_hemoglobina_corregido"):
        st.subheader("📋 Datos del Paciente")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dni = st.text_input("DNI*", max_chars=8, help="8 dígitos")
            nombre_apellido = st.text_input("Nombre completo*")
            edad_meses = st.number_input("Edad (meses)*", min_value=0, max_value=240, value=24)
            peso_kg = st.number_input("Peso (kg)", min_value=0.0, max_value=200.0, value=None, placeholder="Opcional")
            talla_cm = st.number_input("Talla (cm)", min_value=0.0, max_value=200.0, value=None, placeholder="Opcional")
        
        with col2:
            genero = st.selectbox("Género*", ["M", "F", "Otro"])
            region = st.selectbox("Región*", ["LIMA", "AREQUIPA", "CUSCO", "PUNO", "JUNIN", "LA LIBERTAD", "PIURA", "OTRO"])
            hemoglobina_dl1 = st.number_input("Hemoglobina (g/dL)*", min_value=3.0, max_value=20.0, value=12.0, step=0.1)
            interpretacion_hematologica = st.text_area("Interpretación Hematológica", 
                                                      placeholder="Ej: Anemia ferropénica moderada...")
            politicas_de_RLS = st.selectbox("Políticas RLS", 
                                           ["OTRO / NO ESPECIFICADO", "LIMA", "AREQUIPA", "CUSCO", "LAMBAYEQUE"])
        
        # Determinar riesgo automáticamente
        st.subheader("⚡ Determinación Automática de Riesgo")
        
        if hemoglobina_dl1 < 8:
            riesgo = "ALTO RIESGO (Alerta Clínica - ALTA)"
            estado_alerta = "EMERGENCIA"
            sugerencias = "ACCION PRIORITARIA: Derivación inmediata a especialista y suplementación urgente"
        elif hemoglobina_dl1 < 10:
            riesgo = "ALTO RIESGO (Alerta Clínica - MODERADA)"
            estado_alerta = "URGENTE"
            sugerencias = "Suplemento de hierro y control mensual. Evaluar causas secundarias"
        elif hemoglobina_dl1 < 12:
            riesgo = "RIESGO MODERADO"
            estado_alerta = "PRIORITARIO"
            sugerencias = "Dieta rica en hierro y evaluación médica en 2 semanas"
        elif hemoglobina_dl1 < 13:
            riesgo = "BAJO RIESGO"
            estado_alerta = "EN SEGUIMIENTO"
            sugerencias = "PREVENCIÓN: Mantener alimentación balanceada y control anual"
        else:
            riesgo = "NORMAL"
            estado_alerta = "ESTABLE"
            sugerencias = "Valores normales. Mantener hábitos saludables"
        
        # Mostrar resultados automáticos
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.info(f"**Nivel de riesgo:** {riesgo}")
            st.info(f"**Estado de alerta:** {estado_alerta}")
        with col_res2:
            st.info(f"**Sugerencias:** {sugerencias}")
            st.info(f"**Hemoglobina:** {hemoglobina_dl1} g/dL")
        
        # Campos adicionales opcionales
        st.subheader("📝 Información Adicional (Opcional)")
        
        col_add1, col_add2 = st.columns(2)
        with col_add1:
            telefono = st.text_input("Teléfono", placeholder="Opcional")
            departamento = st.text_input("Departamento/Provincia", placeholder="Opcional")
            altitud_msnm = st.number_input("Altitud (msnm)", min_value=0, max_value=5000, value=None)
        
        with col_add2:
            nivel_educativo = st.selectbox("Nivel educativo", 
                                          ["", "Sin educación", "Primaria", "Secundaria", "Superior"])
            acceso_agua_potable = st.checkbox("Acceso a agua potable")
            tiene_servicio_salud = st.checkbox("Tiene servicio de salud")
        
        # Botón de envío
        submitted = st.form_submit_button("💾 Guardar en Supabase (USANDO COLUMNAS CORRECTAS)")
        
        if submitted:
            if not dni or not nombre_apellido:
                st.error("⚠️ Complete DNI y nombre completo")
                return
            
            try:
                # PREPARAR DATOS CON COLUMNAS CORRECTAS
                paciente_data = {
                    'dni': dni,
                    'nombre_apellido': nombre_apellido,
                    'edad_meses': int(edad_meses),
                    'genero': genero,
                    'region': region,
                    'hemoglobina_dl1': float(hemoglobina_dl1),
                    'interpretacion_hematologica': interpretacion_hematologica or "Sin interpretación",
                    'politicas_de_RLS': politicas_de_RLS,
                    'riesgo': riesgo,
                    'estado_alerta': estado_alerta,
                    'sugerencias': sugerencias,
                    'fecha_alerta': datetime.date.today().isoformat(),
                    'estado_paciente': 'Activo'
                }
                
                # Campos opcionales (solo si tienen valor)
                if peso_kg is not None:
                    paciente_data['peso_kg'] = float(peso_kg)
                
                if talla_cm is not None:
                    paciente_data['talla_cm'] = float(talla_cm)
                
                if telefono:
                    paciente_data['telefono'] = telefono
                
                if departamento:
                    paciente_data['departamento'] = departamento
                
                if altitud_msnm is not None:
                    paciente_data['altitud_msnm'] = int(altitud_msnm)
                
                if nivel_educativo:
                    paciente_data['nivel_educativo'] = nivel_educativo
                
                paciente_data['acceso_agua_potable'] = bool(acceso_agua_potable)
                paciente_data['tiene_servicio_salud'] = bool(tiene_servicio_salud)
                
                # Mostrar datos que se enviarán
                with st.expander("🔍 Ver datos a enviar"):
                    st.json(paciente_data)
                    st.write(f"**Total de campos:** {len(paciente_data)}")
                
                # INSERTAR EN SUPABASE
                st.info("🔄 Insertando en Supabase...")
                
                response = supabase.table('alertas_hemoglobina').insert(paciente_data).execute()
                
                if response.data:
                    st.success(f"✅ ¡Paciente {nombre_apellido} registrado exitosamente!")
                    st.balloons()
                    
                    # Mostrar confirmación
                    st.subheader("📋 Resumen del Registro")
                    
                    cols_summary = st.columns(4)
                    with cols_summary[0]:
                        st.metric("DNI", dni)
                    with cols_summary[1]:
                        st.metric("Edad", f"{edad_meses} meses")
                    with cols_summary[2]:
                        st.metric("Hemoglobina", f"{hemoglobina_dl1} g/dL")
                    with cols_summary[3]:
                        st.metric("Riesgo", riesgo.split("(")[0].strip())
                    
                    # Botón para ver datos
                    if st.button("📊 Ver todos los registros"):
                        ver_registros()
                        
                else:
                    st.error("❌ No se recibió respuesta de Supabase")
                    
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Error al guardar: {error_msg}")
                
                # Análisis del error
                if "severidad_interpretacion" in error_msg:
                    st.error("""
                    ❌ **ERROR CRÍTICO:** Estás usando la columna `severidad_interpretacion` que NO existe.
                    
                    **SOLUCIÓN INMEDIATA:**
                    1. Busca en tu código donde dice `severidad_interpretacion`
                    2. Elimínalo o cámbialo por `interpretacion_hematologica`
                    3. Usa SOLO las columnas de la lista de arriba ✅
                    """)
                
                if "interpretacion_hematologica" in error_msg:
                    st.warning("""
                    ℹ️ La columna `interpretacion_hematologica` SÍ existe en tu tabla.
                    Verifica que el nombre esté exactamente igual.
                    """)
                
                # Mostrar estructura de tabla actual
                try:
                    st.info("🔄 Obteniendo estructura actual de la tabla...")
                    test_data = supabase.table('alertas_hemoglobina').select("*").limit(1).execute()
                    if test_data.data:
                        st.write("**Estructura actual (primera fila):**")
                        st.json(test_data.data[0])
                    else:
                        st.write("Tabla vacía o sin datos")
                except:
                    pass

# ============================================
# FUNCIÓN PARA VER REGISTROS EXISTENTES
# ============================================

def ver_registros():
    """Muestra los registros existentes en Supabase"""
    
    st.header("📊 Registros Existentes en Supabase")
    
    if supabase is None:
        st.error("Sin conexión")
        return
    
    try:
        # Obtener datos
        response = supabase.table('alertas_hemoglobina').select("*").order('fecha_alerta', desc=True).execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            st.success(f"✅ {len(df)} registros encontrados")
            
            # Filtrar por región
            if 'region' in df.columns and not df.empty:
                regiones = sorted(df['region'].dropna().unique())
                region_seleccionada = st.selectbox("Filtrar por región:", ["Todas"] + list(regiones))
                
                if region_seleccionada != "Todas":
                    df = df[df['region'] == region_seleccionada]
                    st.info(f"Mostrando {len(df)} registros de {region_seleccionada}")
            
            # Mostrar tabla
            st.dataframe(df, use_container_width=True)
            
            # Estadísticas
            st.subheader("📈 Estadísticas")
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            
            with col_stats1:
                if 'riesgo' in df.columns:
                    alto = df['riesgo'].str.contains('ALTO', case=False, na=False).sum()
                    st.metric("Alertas Alto Riesgo", alto)
            
            with col_stats2:
                if 'hemoglobina_dl1' in df.columns and df['hemoglobina_dl1'].notna().any():
                    promedio_hb = df['hemoglobina_dl1'].mean()
                    st.metric("Hemoglobina Promedio", f"{promedio_hb:.1f} g/dL")
            
            with col_stats3:
                if 'region' in df.columns and not df['region'].mode().empty:
                    region_comun = df['region'].mode().iloc[0]
                    st.metric("Región más común", region_comun)
            
            # Gráficos
            st.subheader("📊 Visualizaciones")
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                if 'riesgo' in df.columns:
                    riesgo_counts = df['riesgo'].value_counts()
                    if not riesgo_counts.empty:
                        fig1 = px.bar(riesgo_counts, 
                                     x=riesgo_counts.index, 
                                     y=riesgo_counts.values,
                                     title='Distribución por Riesgo',
                                     color=riesgo_counts.values,
                                     color_continuous_scale='RdYlGn_r')
                        st.plotly_chart(fig1, use_container_width=True)
            
            with col_chart2:
                if 'region' in df.columns:
                    region_counts = df['region'].value_counts().head(10)
                    if not region_counts.empty:
                        fig2 = px.pie(region_counts, 
                                     names=region_counts.index, 
                                     values=region_counts.values,
                                     title='Top 10 Regiones',
                                     hole=0.3)
                        st.plotly_chart(fig2, use_container_width=True)
            
            # Opción para descargar
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar CSV",
                data=csv,
                file_name=f"hemoglobina_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
        else:
            st.info("📭 No hay registros en la base de datos.")
            
    except Exception as e:
        st.error(f"Error: {e}")

# ============================================
# FUNCIÓN PARA LIMPIAR ERRORES
# ============================================

def limpiar_errores():
    """Ayuda a identificar y corregir errores"""
    
    st.header("🔧 Herramienta de Diagnóstico de Errores")
    
    st.warning("""
    **Error actual:** `'severidad_interpretacion' column not found`
    
    Este error ocurre porque tu código Python está usando nombres de columnas
    que NO existen en tu tabla de Supabase.
    """)
    
    st.subheader("📋 Columnas que DEBES usar:")
    
    columnas_correctas = [
        "dni", "nombre_apellido", "edad_meses", "peso_kg", "talla_cm",
        "genero", "telefono", "estado_paciente", "region", "departamento",
        "altitud_msnm", "nivel_educativo", "acceso_agua_potable",
        "tiene_servicio_salud", "hemoglobina_dl1", "en_seguimiento",
        "consume_hierro", "tipo_suplemento_hierro", "frecuencia_suplemento",
        "antecedentes_anemia", "enfermedades_cronicas", 
        "interpretacion_hematologica", "politicas_de_RLS", "riesgo",
        "fecha_alerta", "estado_alerta", "sugerencias"
    ]
    
    st.write(pd.DataFrame({"Columnas Correctas": columnas_correctas}))
    
    st.subheader("🔍 Buscar errores en tu código:")
    
    codigo_analizar = st.text_area("Pega tu código Python aquí para analizar:")
    
    if codigo_analizar:
        errores = []
        
        # Buscar columnas problemáticas
        columnas_problematicas = [
            "severidad_interpretacion",
            "nombre_apellide",
            "regién",
            "interpretacion",  # Posible abreviatura
            "severidad",
            "severity",
            "interpretation"
        ]
        
        for col_problema in columnas_problematicas:
            if col_problema in codigo_analizar.lower():
                errores.append(f"❌ Posible columna errónea: '{col_problema}'")
        
        if errores:
            st.error("**Errores encontrados:**")
            for error in errores:
                st.write(error)
            
            st.success("**Solución:** Reemplaza con columnas de la lista de arriba ✅")
        else:
            st.success("✅ No se encontraron columnas problemáticas evidentes")

# ============================================
# MAIN APP
# ============================================

# Título principal
st.sidebar.title("🏥 Sistema de Hemoglobina")

# Menú de navegación
opcion = st.sidebar.selectbox(
    "Seleccione opción:",
    [
        "🩸 Registrar Hemoglobina",
        "📊 Ver Registros", 
        "🔧 Diagnosticar Errores",
        "ℹ️ Información"
    ]
)

# Información de conexión
st.sidebar.markdown("---")
st.sidebar.subheader("🔗 Estado de Conexión")

if supabase:
    st.sidebar.success("✅ Conectado a Supabase")
    
    try:
        # Contar registros
        response = supabase.table('alertas_hemoglobina').select("dni", count='exact').execute()
        if hasattr(response, 'count'):
            st.sidebar.info(f"📊 {response.count} registros")
        elif response.data:
            st.sidebar.info(f"📊 {len(response.data)} registros")
    except:
        st.sidebar.warning("⚠️ Error al contar")
else:
    st.sidebar.error("❌ Sin conexión")

# Mostrar módulo seleccionado
if opcion == "🩸 Registrar Hemoglobina":
    registrar_hemoglobina_corregido()
    
elif opcion == "📊 Ver Registros":
    ver_registros()
    
elif opcion == "🔧 Diagnosticar Errores":
    limpiar_errores()
    
elif opcion == "ℹ️ Información":
    st.title("ℹ️ Información del Sistema")
    
    st.info("""
    **✅ ESTA VERSIÓN ESTÁ CORREGIDA**
    
    **Problema anterior:** Usaba columnas que NO existen (`severidad_interpretacion`)
    
    **Solución actual:** Usa SOLO columnas que SÍ existen en tu tabla Supabase.
    
    **Columnas principales usadas:**
    - `dni` - Identificación
    - `nombre_apellido` - Nombre completo
    - `hemoglobina_dl1` - Nivel de hemoglobina
    - `riesgo` - Nivel de riesgo (calculado automáticamente)
    - `interpretacion_hematologica` - Interpretación del médico
    - `region` - Región del paciente
    
    **Credenciales Supabase:**
    - URL: `https://kwsuszkblbejvliniggd.supabase.co`
    - Tabla: `alertas_hemoglobina`
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("v2.0 - Corregido ✅")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 20px;">
    <p>🏥 <strong>Sistema de Monitoreo de Hemoglobina</strong></p>
    <p>✅ <strong>VERSIÓN CORREGIDA</strong> - Usando columnas reales</p>
    <p>Ministerio de Salud - Todos los derechos reservados</p>
</div>
""", unsafe_allow_html=True)
