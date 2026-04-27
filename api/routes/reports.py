from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import settings

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/{report_id}/download")
async def download_report(report_id: str):
    """Downloads the PDF for a previously generated report."""
    pdf_path = settings.reports_output_dir / f"{report_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=f"ergo_report_{report_id}.pdf",
    )
