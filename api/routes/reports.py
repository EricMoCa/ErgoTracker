from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import settings

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/{report_id}/download")
async def download_report(report_id: str):
    """Descarga el PDF del reporte."""
    pdf_path = settings.reports_output_dir / f"{report_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Reporte no encontrado")
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=f"ergo_report_{report_id}.pdf",
    )


@router.get("/{report_id}/video")
async def stream_annotated_video(report_id: str):
    """Devuelve el MP4 con el esqueleto 3D proyectado sobre el video original."""
    video_path = settings.reports_output_dir / f"{report_id}_annotated.mp4"
    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Video anotado no disponible. El análisis puede seguir procesando o falló la generación.",
        )
    return FileResponse(
        str(video_path),
        media_type="video/mp4",
        filename=f"ergo_annotated_{report_id}.mp4",
    )
