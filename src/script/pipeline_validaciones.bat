@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  PIPELINE VALIDACIONES
echo ============================================================
echo.
echo  Ejecuta las validaciones y caracterizaciones que justifican
echo  las decisiones del pipeline principal.
echo.
echo  Los aeropuertos se leen de config.AEROPUERTOS_VALIDACION.
echo.
echo  REQUIERE haber ejecutado pipeline_ejecutar.bat antes.
echo.

cd /d "%~dp0\..\.."

if not exist "resultados\macro\clusters_macro.parquet" (
    echo ERROR: No se encuentran resultados macro.
    echo Ejecuta primero pipeline_ejecutar.bat
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  [LIMPIEZA] Borrando resultados de validaciones anteriores
echo ============================================================

if exist "resultados\validaciones" rmdir /S /Q "resultados\validaciones"
echo Resultados de validaciones eliminados.

echo.
echo ============================================================
echo  CARACTERIZACION
echo ============================================================

echo Caracterizacion macro...
python -m src.macro.caracterizacion_macro
if !ERRORLEVEL! NEQ 0 goto error

echo Visualizacion macro (figura PNG)...
python -m src.macro.visualizar_macro
if !ERRORLEVEL! NEQ 0 goto error

echo Caracterizacion micro WED (todos los aeropuertos del pipeline)...
python -m src.micro.caracterizacion_micro --esquema WED
if !ERRORLEVEL! NEQ 0 goto error

echo.
echo ============================================================
echo  VALIDACION METERING FIXES (aeropuertos de AEROPUERTOS_VALIDACION)
echo ============================================================

echo Validacion contra metering fixes WED...
python -m src.validaciones.validacion_metering_fixes --esquema WED --aeropuertos LOWW EKCH
if !ERRORLEVEL! NEQ 0 goto error

echo Flujos medios WED...
python -m src.validaciones.flujos_medios --esquema WED --aeropuertos LOWW EKCH
if !ERRORLEVEL! NEQ 0 goto error

echo.
echo ============================================================
echo  COMPARACION ED vs WED (aeropuertos de AEROPUERTOS_VALIDACION)
echo ============================================================
for /f "delims=" %%A in ('python -c "from src import config; [print(a) for a in config.AEROPUERTOS_VALIDACION]"') do (
    echo.
    echo --- Distancias micro %%A ED ---
    python -m src.micro.distancias_micro %%A --esquema ED
    if !ERRORLEVEL! NEQ 0 goto error

    echo --- Clustering micro %%A ED ---
    python -m src.micro.clustering_micro %%A --esquema ED
    if !ERRORLEVEL! NEQ 0 goto error

    echo --- Figura comparacion ED vs WED %%A ---
    python -m src.validaciones.comparacion_ed_wed %%A
    if !ERRORLEVEL! NEQ 0 goto error
)

echo.
echo ============================================================
echo  ANALISIS COMPLEMENTARIO
echo ============================================================

echo Analisis del ruido macro...
python -m src.validaciones.analisis_ruido_macro
if !ERRORLEVEL! NEQ 0 goto error

echo Figura de todos los flujos LOWW...
python -m src.validaciones.flujos_todos_loww
if !ERRORLEVEL! NEQ 0 goto error

echo Diagnostico de vuelos largos...
python -m src.validaciones.diagnostico_vuelos_largos
if !ERRORLEVEL! NEQ 0 goto error

echo.
echo ============================================================
echo  VALIDACIONES COMPLETADAS
echo ============================================================
echo.
echo  Resultados en: resultados\validaciones\
echo  Logs en:       resultados\logs\
echo.
pause
exit /b 0

:error
echo.
echo ============================================================
echo  ERROR EN EL ULTIMO PASO EJECUTADO
echo ============================================================
pause
exit /b 1