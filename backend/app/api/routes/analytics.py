"""
CSV Analytics API Routes
"""

import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException

from app.schemas.requests import CSVAnalyticsRequest
from app.schemas.responses import CSVAnalyticsResponse
from app.services.csv_analytics_service import CSVAnalyticsService, AnalyticsRequest
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analytics", tags=["analytics"])

# Initialize service
csv_analytics_service = CSVAnalyticsService()


@router.post("/csv", response_model=CSVAnalyticsResponse)
async def analyze_csv_data(request: CSVAnalyticsRequest):
    """
    Perform analytics on CSV data using natural language queries.
    
    Args:
        request: CSV analytics request
    """
    try:
        # Find CSV file in dataset
        dataset_dir = Path(settings.datasets_directory) / request.dataset_name
        if not dataset_dir.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        csv_files = list(dataset_dir.glob("*.csv"))
        if not csv_files:
            raise HTTPException(status_code=404, detail="No CSV files found in dataset")
        
        # Use specified file or first CSV found
        if request.file_name:
            csv_path = dataset_dir / request.file_name
            if not csv_path.exists():
                raise HTTPException(status_code=404, detail="CSV file not found")
        else:
            csv_path = csv_files[0]
        
        # Create analytics request
        analytics_request = AnalyticsRequest(
            data=str(csv_path),
            query=request.query,
            filters=request.filters,
            visualize=request.visualize
        )
        
        # Perform analysis
        response = await csv_analytics_service.analyze_csv(analytics_request)
        
        # Export if requested
        if request.export_format:
            await csv_analytics_service.export_results(
                response,
                request.export_format
            )
        
        return CSVAnalyticsResponse(
            status="success",
            analysis_results=response.analysis_results,
            summary=response.natural_language_summary,
            data_preview=response.data_preview,
            visualizations=response.visualizations,
            statistics=response.statistics,
            query_sql=response.query_sql,
            processing_time=response.processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CSV analytics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis-types")
async def get_analysis_types():
    """Get available analysis types for CSV data."""
    return {
        "analysis_types": [
            "descriptive",
            "correlation",
            "trend",
            "distribution",
            "comparison",
            "aggregation",
            "custom"
        ],
        "chart_types": [
            "line",
            "bar",
            "scatter",
            "histogram",
            "pie",
            "heatmap",
            "box",
            "area"
        ],
        "export_formats": ["json", "csv", "html"],
        "sample_queries": [
            "What is the average sales by region?",
            "Show me the trend of revenue over time",
            "Compare performance across different categories",
            "What are the correlations between variables?",
            "Find outliers in the data",
            "Group by month and calculate totals"
        ]
    }