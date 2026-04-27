# ErgoTracker — Guía de Orquestación de Agentes

## Orden de implementación

```
Fase 1 (secuencial):  schemas/
Fase 2 (paralelo):    pose_pipeline/ | ergo_engine/ | llm_rules/ | reports/
Fase 3 (secuencial):  advanced_pipeline/   ← espera pose_pipeline
Fase 4 (secuencial):  api/                 ← espera todos
```

## Comando principal (pegar en Claude Code)

Abrir Claude Code en la raíz del proyecto y ejecutar:

```
claude
```

Luego pegar este prompt:

---

## PROMPT DE ORQUESTACIÓN (copiar y pegar en Claude Code)

```
Eres el orquestador de implementación de ErgoTracker.
El proyecto está en el directorio actual.

Sigue este orden estricto:

FASE 1 — schemas/ (todos dependen de este):
Usa un subagente para implementar schemas/ siguiendo schemas/CLAUDE.md exactamente.
Implementar en orden: video.py → skeleton.py → angles.py → ergonomic.py → __init__.py
Al terminar, correr: pytest schemas/ -v
No continuar a Fase 2 hasta que todos los tests pasen.

FASE 2 — Implementación paralela (4 subagentes simultáneos):
Una vez schemas/ esté completo, lanzar SIMULTÁNEAMENTE estos 4 subagentes:

  Subagente A: implementar pose_pipeline/ siguiendo pose_pipeline/CLAUDE.md
    Orden: model_downloader.py → detector.py → pose_2d.py → pose_3d.py → height.py → pipeline.py
    Al terminar: pytest pose_pipeline/ -v

  Subagente B: implementar ergo_engine/ siguiendo ergo_engine/CLAUDE.md
    Orden: angles.py → reba.py → rula.py → owas.py → llm_rule_analyzer.py → engine.py
    Al terminar: pytest ergo_engine/ -v

  Subagente C: implementar llm_rules/ siguiendo llm_rules/CLAUDE.md
    Orden: gpu_manager.py → pdf_extractor.py → rule_cache.py → rule_extractor.py
    Al terminar: pytest llm_rules/ -v

  Subagente D: implementar reports/ siguiendo reports/CLAUDE.md
    Orden: charts.py → templates/report_template.html → generator.py
    Al terminar: pytest reports/ -v

Esperar a que los 4 subagentes terminen antes de continuar.

FASE 3 — advanced_pipeline/ (espera pose_pipeline):
Una vez Subagente A (pose_pipeline) haya terminado, implementar advanced_pipeline/
siguiendo advanced_pipeline/CLAUDE.md.
Orden: visual_odometry.py → pipeline_router.py → contact_refinement.py →
       stride_refinement.py → gvhmr_wrapper.py → wham_wrapper.py → tram_wrapper.py →
       humanmm_wrapper.py → pipeline.py
Al terminar: pytest advanced_pipeline/ -v

FASE 4 — api/ (espera todos):
Una vez Fases 1-3 completas, implementar api/ siguiendo api/CLAUDE.md.
Orden: config.py → storage/job_store.py → services/orchestrator.py →
       routes/analysis.py → routes/rules.py → routes/reports.py → main.py
Al terminar: pytest api/ -v

FASE FINAL — Verificación global:
pytest --tb=short
Si algún test falla, corregir antes de reportar como completado.
```

---

## Comandos por módulo (para lanzar individualmente en terminales separadas)

Si prefieres lanzar cada módulo manualmente en terminales paralelas:

### Terminal 1 — schemas (primero siempre)
```bash
cd C:\WIP\ClaudeProjects\ErgoTracker
claude "Implementa schemas/ siguiendo schemas/CLAUDE.md exactamente. Orden: video.py, skeleton.py, angles.py, ergonomic.py. Corre pytest schemas/ -v al terminar y corrige errores."
```

### Terminal 2 — pose_pipeline (después de schemas)
```bash
cd C:\WIP\ClaudeProjects\ErgoTracker
claude "Implementa pose_pipeline/ siguiendo pose_pipeline/CLAUDE.md. Schemas ya está completo. Orden: model_downloader.py, detector.py, pose_2d.py, pose_3d.py, height.py, pipeline.py. Corre pytest pose_pipeline/ -v al terminar."
```

### Terminal 3 — ergo_engine (después de schemas, paralelo con T2)
```bash
cd C:\WIP\ClaudeProjects\ErgoTracker
claude "Implementa ergo_engine/ siguiendo ergo_engine/CLAUDE.md. Schemas ya está completo. Orden: angles.py, reba.py, rula.py, owas.py, llm_rule_analyzer.py, engine.py. Corre pytest ergo_engine/ -v al terminar."
```

### Terminal 4 — llm_rules (después de schemas, paralelo con T2 y T3)
```bash
cd C:\WIP\ClaudeProjects\ErgoTracker
claude "Implementa llm_rules/ siguiendo llm_rules/CLAUDE.md. Schemas ya está completo. Orden: gpu_manager.py, pdf_extractor.py, rule_cache.py, rule_extractor.py. Corre pytest llm_rules/ -v al terminar."
```

### Terminal 5 — reports (después de schemas, paralelo con T2, T3, T4)
```bash
cd C:\WIP\ClaudeProjects\ErgoTracker
claude "Implementa reports/ siguiendo reports/CLAUDE.md. Schemas ya está completo. Orden: charts.py, templates/report_template.html, generator.py. Corre pytest reports/ -v al terminar."
```

### Terminal 6 — advanced_pipeline (después de pose_pipeline)
```bash
cd C:\WIP\ClaudeProjects\ErgoTracker
claude "Implementa advanced_pipeline/ siguiendo advanced_pipeline/CLAUDE.md. Schemas y pose_pipeline ya están completos. Orden: visual_odometry.py, pipeline_router.py, contact_refinement.py, stride_refinement.py, gvhmr_wrapper.py, wham_wrapper.py, tram_wrapper.py, humanmm_wrapper.py, pipeline.py. Corre pytest advanced_pipeline/ -v al terminar."
```

### Terminal 7 — api (después de todos)
```bash
cd C:\WIP\ClaudeProjects\ErgoTracker
claude "Implementa api/ siguiendo api/CLAUDE.md. Todos los módulos previos ya están completos. Orden: config.py, storage/job_store.py, services/orchestrator.py, routes/analysis.py, routes/rules.py, routes/reports.py, main.py. Corre pytest api/ -v al terminar."
```

---

## Dependencias a instalar antes de empezar

```bash
# Base (todos los módulos)
pip install -r requirements/base.txt

# Por módulo (instalar en la terminal correspondiente)
pip install -r requirements/pose.txt      # Terminal 2
pip install -r requirements/ergo.txt      # Terminal 3
pip install -r requirements/llm.txt       # Terminal 4
pip install -r requirements/reports.txt   # Terminal 5
pip install -r requirements/advanced.txt  # Terminal 6
pip install -r requirements/api.txt       # Terminal 7

# Modelos avanzados (opcional, para advanced_pipeline)
# git clone https://github.com/zju3dv/GVHMR && pip install -e GVHMR/
# git clone https://github.com/yufu-wang/tram && pip install -e tram/
```

## Notas importantes para los agentes

- Cada módulo tiene su `CLAUDE.md` con la implementación completa. Seguirlo exactamente.
- Los modelos avanzados (GVHMR, WHAM, TRAM) deben implementarse como stubs que retornan
  `None` si no están instalados. El sistema siempre tiene fallback a MotionBERT Lite (CPU).
- La REGLA DE ORO es: ningún módulo importa de otro módulo excepto de `schemas/`.
  La única excepción es `advanced_pipeline/` que puede importar de `pose_pipeline/`.
- Si un test falla, corregir antes de reportar como completado.
