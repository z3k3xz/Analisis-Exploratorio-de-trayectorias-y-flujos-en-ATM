@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  PIPELINE ABLACIONES
echo ============================================================
echo.
echo  Ejecuta los estudios de sensibilidad sobre los
echo  hiperparametros del pipeline.
echo.
echo  Los aeropuertos micro se leen de config.AEROPUERTOS_VALIDACION.
echo.
echo  REQUIERE haber ejecutado pipeline_ejecutar.bat antes.
echo.

cd /d "%~dp0\..\.."

if not exist "resultados\macro\matriz_distancias_macro.npy" (
    echo ERROR: No se encuentran resultados macro.
    echo Ejecuta primero pipeline_ejecutar.bat
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  [LIMPIEZA] Borrando resultados de ablaciones anteriores
echo ============================================================

if exist "resultados\validaciones\ablacion_mcs_macro*" del /Q "resultados\validaciones\ablacion_mcs_macro*"
if exist "resultados\validaciones\ablacion_n_puntos*" del /Q "resultados\validaciones\ablacion_n_puntos*"
for /f "delims=" %%A in ('python -c "from src import config; [print(a) for a in config.AEROPUERTOS_VALIDACION]"') do (
    if exist "resultados\validaciones\ablacion_mcs_micro_%%A*" del /Q "resultados\validaciones\ablacion_mcs_micro_%%A*"
)
echo Resultados de ablaciones eliminados.

echo.
echo ============================================================
echo  ABLACION min_cluster_size MACRO
echo ============================================================

python -m src.validaciones.ablacion_min_cluster --nivel macro
if !ERRORLEVEL! NEQ 0 goto error

echo.
echo ============================================================
echo  ABLACION min_cluster_size MICRO WED
echo ============================================================
for /f "delims=" %%A in ('python -c "from src import config; [print(a) for a in config.AEROPUERTOS_VALIDACION]"') do (
    echo.
    echo --- Ablacion min_cluster_size %%A WED ---
    python -m src.validaciones.ablacion_min_cluster --nivel micro --icao %%A --esquema WED
    if !ERRORLEVEL! NEQ 0 goto error
)

echo.
echo ============================================================
echo  ABLACION N PUNTOS DE REMUESTREO MACRO
echo ============================================================

python -m src.validaciones.ablacion_n_puntos
if !ERRORLEVEL! NEQ 0 goto error

echo.
echo ============================================================
echo  ABLACIONES COMPLETADAS
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