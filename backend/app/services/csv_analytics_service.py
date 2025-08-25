"""
CSV Analytics Service

This module provides advanced CSV and data analytics capabilities including:
- Statistical analysis and aggregations
- Data visualization generation
- Natural language query processing for data
- Data transformations and filtering
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import json
from datetime import datetime
import io
import base64
from dataclasses import dataclass
from enum import Enum

# For data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

from ..services.ai_providers import get_llm_provider
from ..core.config import settings

logger = logging.getLogger(__name__)

# Configure matplotlib for server environment
plt.switch_backend('Agg')
sns.set_style("whitegrid")


class AnalysisType(str, Enum):
    """Types of analysis available"""
    DESCRIPTIVE = "descriptive"
    CORRELATION = "correlation"
    TREND = "trend"
    DISTRIBUTION = "distribution"
    COMPARISON = "comparison"
    AGGREGATION = "aggregation"
    CUSTOM = "custom"


class ChartType(str, Enum):
    """Types of charts available"""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    PIE = "pie"
    HEATMAP = "heatmap"
    BOX = "box"
    AREA = "area"


@dataclass
class AnalyticsRequest:
    """Analytics request configuration"""
    data: Union[pd.DataFrame, str]  # DataFrame or file path
    query: str
    analysis_type: Optional[AnalysisType] = None
    columns: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    aggregations: Optional[Dict[str, str]] = None
    visualize: bool = True
    chart_type: Optional[ChartType] = None


@dataclass
class AnalyticsResponse:
    """Analytics response with results and visualizations"""
    analysis_results: Dict[str, Any]
    natural_language_summary: str
    data_preview: Dict[str, Any]
    visualizations: Optional[List[Dict[str, Any]]] = None
    statistics: Optional[Dict[str, Any]] = None
    query_sql: Optional[str] = None
    processing_time: float = 0


class CSVAnalyticsService:
    """Advanced CSV and data analytics service"""
    
    def __init__(self):
        self.llm_provider = get_llm_provider()
        self.current_df: Optional[pd.DataFrame] = None
        self.analysis_cache = {}
    
    async def analyze_csv(self, request: AnalyticsRequest) -> AnalyticsResponse:
        """Perform analytics on CSV data based on natural language query"""
        start_time = datetime.now()
        
        try:
            # Load data if path provided
            if isinstance(request.data, str):
                df = pd.read_csv(request.data)
            else:
                df = request.data
            
            self.current_df = df
            
            # Parse natural language query to determine analysis
            analysis_plan = await self._parse_query(request.query, df)
            
            # Apply filters if specified
            if request.filters or analysis_plan.get("filters"):
                df = self._apply_filters(df, request.filters or analysis_plan.get("filters"))
            
            # Perform requested analysis
            analysis_results = await self._perform_analysis(
                df,
                analysis_plan,
                request.analysis_type
            )
            
            # Generate visualizations if requested
            visualizations = None
            if request.visualize:
                visualizations = await self._generate_visualizations(
                    df,
                    analysis_results,
                    analysis_plan,
                    request.chart_type
                )
            
            # Generate statistics
            statistics = self._generate_statistics(df, analysis_plan.get("columns"))
            
            # Generate natural language summary
            summary = await self._generate_summary(
                request.query,
                analysis_results,
                statistics
            )
            
            # Create data preview
            data_preview = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "head": df.head(10).to_dict('records'),
                "sample": df.sample(min(5, len(df))).to_dict('records') if len(df) > 5 else None
            }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AnalyticsResponse(
                analysis_results=analysis_results,
                natural_language_summary=summary,
                data_preview=data_preview,
                visualizations=visualizations,
                statistics=statistics,
                query_sql=analysis_plan.get("sql"),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error analyzing CSV: {str(e)}")
            raise
    
    async def _parse_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse natural language query to determine analysis requirements"""
        
        columns_info = {
            col: str(df[col].dtype) for col in df.columns
        }
        
        prompt = f"""
        Parse this natural language query about data analysis into a structured plan.
        
        Query: {query}
        
        Available columns and types:
        {json.dumps(columns_info, indent=2)}
        
        Data shape: {df.shape}
        
        Return a JSON object with:
        {{
            "analysis_type": "descriptive|correlation|trend|distribution|comparison|aggregation",
            "columns": ["column1", "column2"],
            "operations": ["mean", "sum", "count", "groupby", etc.],
            "filters": {{"column": "condition"}},
            "group_by": ["column"],
            "order_by": {{"column": "asc|desc"}},
            "chart_type": "line|bar|scatter|histogram|pie|heatmap|box",
            "sql": "equivalent SQL query if applicable"
        }}
        """
        
        response = await self.llm_provider.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=500
        )
        
        try:
            # Parse LLM response as JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback to basic analysis
                return {
                    "analysis_type": "descriptive",
                    "columns": df.columns.tolist()[:5],
                    "operations": ["describe"]
                }
        except:
            return {
                "analysis_type": "descriptive",
                "columns": df.columns.tolist()[:5],
                "operations": ["describe"]
            }
    
    async def _perform_analysis(
        self,
        df: pd.DataFrame,
        analysis_plan: Dict[str, Any],
        analysis_type: Optional[AnalysisType] = None
    ) -> Dict[str, Any]:
        """Perform the requested analysis on the data"""
        
        results = {}
        analysis_type = analysis_type or analysis_plan.get("analysis_type", "descriptive")
        
        if analysis_type == "descriptive" or analysis_type == AnalysisType.DESCRIPTIVE:
            results = self._descriptive_analysis(df, analysis_plan.get("columns"))
            
        elif analysis_type == "correlation" or analysis_type == AnalysisType.CORRELATION:
            results = self._correlation_analysis(df, analysis_plan.get("columns"))
            
        elif analysis_type == "trend" or analysis_type == AnalysisType.TREND:
            results = self._trend_analysis(df, analysis_plan)
            
        elif analysis_type == "distribution" or analysis_type == AnalysisType.DISTRIBUTION:
            results = self._distribution_analysis(df, analysis_plan.get("columns"))
            
        elif analysis_type == "comparison" or analysis_type == AnalysisType.COMPARISON:
            results = self._comparison_analysis(df, analysis_plan)
            
        elif analysis_type == "aggregation" or analysis_type == AnalysisType.AGGREGATION:
            results = self._aggregation_analysis(df, analysis_plan)
            
        else:
            # Custom analysis based on operations
            results = self._custom_analysis(df, analysis_plan)
        
        return results
    
    def _descriptive_analysis(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform descriptive statistical analysis"""
        if columns:
            df_subset = df[columns].select_dtypes(include=[np.number])
        else:
            df_subset = df.select_dtypes(include=[np.number])
        
        if df_subset.empty:
            return {"message": "No numeric columns found for analysis"}
        
        description = df_subset.describe().to_dict()
        
        return {
            "summary_statistics": description,
            "missing_values": df_subset.isnull().sum().to_dict(),
            "unique_counts": {col: df[col].nunique() for col in df_subset.columns},
            "data_types": df_subset.dtypes.astype(str).to_dict()
        }
    
    def _correlation_analysis(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform correlation analysis"""
        if columns:
            numeric_cols = df[columns].select_dtypes(include=[np.number]).columns
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {"message": "Need at least 2 numeric columns for correlation"}
        
        corr_matrix = df[numeric_cols].corr()
        
        # Find strongest correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Threshold for strong correlation
                    strong_corr.append({
                        "var1": corr_matrix.columns[i],
                        "var2": corr_matrix.columns[j],
                        "correlation": round(corr_value, 3)
                    })
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": strong_corr,
            "variables_analyzed": numeric_cols.tolist()
        }
    
    def _trend_analysis(self, df: pd.DataFrame, analysis_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Perform trend analysis over time"""
        # Identify date column
        date_cols = df.select_dtypes(include=['datetime64', 'object']).columns
        date_col = None
        
        for col in date_cols:
            try:
                pd.to_datetime(df[col])
                date_col = col
                break
            except:
                continue
        
        if not date_col:
            return {"message": "No date column found for trend analysis"}
        
        df[date_col] = pd.to_datetime(df[date_col])
        df_sorted = df.sort_values(date_col)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]  # Analyze top 3 numeric columns
        
        trends = {}
        for col in numeric_cols:
            # Calculate moving average
            if len(df) > 7:
                ma7 = df_sorted[col].rolling(window=7, min_periods=1).mean()
                trends[col] = {
                    "current": df_sorted[col].iloc[-1] if len(df_sorted) > 0 else None,
                    "mean": df_sorted[col].mean(),
                    "change": df_sorted[col].iloc[-1] - df_sorted[col].iloc[0] if len(df_sorted) > 1 else 0,
                    "percent_change": ((df_sorted[col].iloc[-1] / df_sorted[col].iloc[0]) - 1) * 100 if len(df_sorted) > 1 and df_sorted[col].iloc[0] != 0 else 0,
                    "trend_direction": "increasing" if df_sorted[col].iloc[-1] > df_sorted[col].iloc[0] else "decreasing"
                }
        
        return {
            "time_column": date_col,
            "trends": trends,
            "time_range": {
                "start": str(df_sorted[date_col].min()),
                "end": str(df_sorted[date_col].max())
            }
        }
    
    def _distribution_analysis(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze data distribution"""
        if columns:
            numeric_cols = df[columns].select_dtypes(include=[np.number]).columns
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]  # Limit to 5 columns
        
        distributions = {}
        for col in numeric_cols:
            distributions[col] = {
                "mean": df[col].mean(),
                "median": df[col].median(),
                "std": df[col].std(),
                "skewness": df[col].skew(),
                "kurtosis": df[col].kurtosis(),
                "quartiles": {
                    "q1": df[col].quantile(0.25),
                    "q2": df[col].quantile(0.50),
                    "q3": df[col].quantile(0.75)
                },
                "outliers": len(df[(df[col] < df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))) | 
                                  (df[col] > df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))])
            }
        
        return {"distributions": distributions}
    
    def _comparison_analysis(self, df: pd.DataFrame, analysis_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparison analysis between groups"""
        group_by = analysis_plan.get("group_by", [])
        if not group_by:
            # Try to find a categorical column
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                group_by = [cat_cols[0]]
            else:
                return {"message": "No categorical columns found for comparison"}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]
        
        comparisons = {}
        for col in numeric_cols:
            grouped = df.groupby(group_by[0])[col].agg(['mean', 'sum', 'count', 'std'])
            comparisons[col] = grouped.to_dict('index')
        
        return {
            "grouped_by": group_by[0],
            "comparisons": comparisons
        }
    
    def _aggregation_analysis(self, df: pd.DataFrame, analysis_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Perform aggregation analysis"""
        group_by = analysis_plan.get("group_by", [])
        aggregations = analysis_plan.get("aggregations", {})
        
        if not group_by:
            # Simple aggregations without grouping
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            results = {}
            for col in numeric_cols:
                results[col] = {
                    "sum": df[col].sum(),
                    "mean": df[col].mean(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "count": df[col].count()
                }
            return {"aggregations": results}
        
        # Group by and aggregate
        agg_dict = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            agg_dict[col] = ['sum', 'mean', 'count']
        
        grouped = df.groupby(group_by).agg(agg_dict)
        
        return {
            "grouped_by": group_by,
            "aggregations": grouped.to_dict()
        }
    
    def _custom_analysis(self, df: pd.DataFrame, analysis_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Perform custom analysis based on operations"""
        results = {}
        operations = analysis_plan.get("operations", [])
        
        for op in operations:
            if op == "describe":
                results["description"] = df.describe().to_dict()
            elif op == "info":
                buffer = io.StringIO()
                df.info(buf=buffer)
                results["info"] = buffer.getvalue()
            elif op == "value_counts":
                for col in df.select_dtypes(include=['object', 'category']).columns[:3]:
                    results[f"{col}_value_counts"] = df[col].value_counts().head(10).to_dict()
            elif op == "null_analysis":
                results["null_counts"] = df.isnull().sum().to_dict()
                results["null_percentage"] = (df.isnull().sum() / len(df) * 100).to_dict()
        
        return results
    
    async def _generate_visualizations(
        self,
        df: pd.DataFrame,
        analysis_results: Dict[str, Any],
        analysis_plan: Dict[str, Any],
        chart_type: Optional[ChartType] = None
    ) -> List[Dict[str, Any]]:
        """Generate appropriate visualizations for the analysis"""
        visualizations = []
        
        chart_type = chart_type or analysis_plan.get("chart_type", "auto")
        
        # Determine appropriate charts based on analysis type
        if analysis_plan.get("analysis_type") == "correlation":
            viz = self._create_heatmap(df, analysis_results)
            if viz:
                visualizations.append(viz)
        
        elif analysis_plan.get("analysis_type") == "trend":
            viz = self._create_line_chart(df, analysis_results)
            if viz:
                visualizations.append(viz)
        
        elif analysis_plan.get("analysis_type") == "distribution":
            viz = self._create_histogram(df, analysis_results)
            if viz:
                visualizations.append(viz)
            viz = self._create_box_plot(df, analysis_results)
            if viz:
                visualizations.append(viz)
        
        elif analysis_plan.get("analysis_type") == "comparison":
            viz = self._create_bar_chart(df, analysis_results)
            if viz:
                visualizations.append(viz)
        
        else:
            # Auto-generate appropriate charts
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                viz = self._create_scatter_plot(df, numeric_cols[:2])
                if viz:
                    visualizations.append(viz)
            
            if len(numeric_cols) >= 1:
                viz = self._create_histogram(df, {"distributions": {numeric_cols[0]: {}}})
                if viz:
                    visualizations.append(viz)
        
        return visualizations
    
    def _create_heatmap(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create correlation heatmap"""
        try:
            if "correlation_matrix" not in analysis_results:
                return None
            
            corr_data = analysis_results["correlation_matrix"]
            corr_df = pd.DataFrame(corr_data)
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_df.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Correlation Matrix",
                xaxis_title="Variables",
                yaxis_title="Variables",
                width=600,
                height=500
            )
            
            return {
                "type": "heatmap",
                "title": "Correlation Matrix",
                "data": json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))
            }
        except Exception as e:
            logger.error(f"Error creating heatmap: {str(e)}")
            return None
    
    def _create_line_chart(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create line chart for trends"""
        try:
            if "trends" not in analysis_results:
                return None
            
            time_col = analysis_results.get("time_column")
            if not time_col:
                return None
            
            fig = go.Figure()
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]
            for col in numeric_cols:
                fig.add_trace(go.Scatter(
                    x=df[time_col],
                    y=df[col],
                    mode='lines+markers',
                    name=col
                ))
            
            fig.update_layout(
                title="Trend Analysis",
                xaxis_title=time_col,
                yaxis_title="Values",
                hovermode='x unified',
                width=800,
                height=400
            )
            
            return {
                "type": "line",
                "title": "Trend Analysis",
                "data": json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))
            }
        except Exception as e:
            logger.error(f"Error creating line chart: {str(e)}")
            return None
    
    def _create_bar_chart(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create bar chart for comparisons"""
        try:
            if "comparisons" not in analysis_results:
                return None
            
            comparisons = analysis_results["comparisons"]
            if not comparisons:
                return None
            
            # Get first numeric column's comparison
            first_col = list(comparisons.keys())[0]
            data = comparisons[first_col]
            
            categories = list(data.keys())
            values = [data[cat].get('mean', 0) for cat in categories]
            
            fig = go.Figure(data=[
                go.Bar(x=categories, y=values, name=first_col)
            ])
            
            fig.update_layout(
                title=f"Comparison of {first_col}",
                xaxis_title=analysis_results.get("grouped_by", "Category"),
                yaxis_title=first_col,
                width=700,
                height=400
            )
            
            return {
                "type": "bar",
                "title": f"Comparison of {first_col}",
                "data": json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))
            }
        except Exception as e:
            logger.error(f"Error creating bar chart: {str(e)}")
            return None
    
    def _create_histogram(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create histogram for distributions"""
        try:
            distributions = analysis_results.get("distributions", {})
            if not distributions:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    return None
                col = numeric_cols[0]
            else:
                col = list(distributions.keys())[0]
            
            fig = go.Figure(data=[
                go.Histogram(x=df[col], nbinsx=30, name=col)
            ])
            
            fig.update_layout(
                title=f"Distribution of {col}",
                xaxis_title=col,
                yaxis_title="Frequency",
                showlegend=False,
                width=600,
                height=400
            )
            
            return {
                "type": "histogram",
                "title": f"Distribution of {col}",
                "data": json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))
            }
        except Exception as e:
            logger.error(f"Error creating histogram: {str(e)}")
            return None
    
    def _create_box_plot(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create box plot for distributions"""
        try:
            distributions = analysis_results.get("distributions", {})
            numeric_cols = list(distributions.keys()) if distributions else df.select_dtypes(include=[np.number]).columns[:5]
            
            if len(numeric_cols) == 0:
                return None
            
            fig = go.Figure()
            
            for col in numeric_cols:
                fig.add_trace(go.Box(y=df[col], name=col))
            
            fig.update_layout(
                title="Distribution Comparison",
                yaxis_title="Values",
                showlegend=True,
                width=700,
                height=400
            )
            
            return {
                "type": "box",
                "title": "Distribution Comparison",
                "data": json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))
            }
        except Exception as e:
            logger.error(f"Error creating box plot: {str(e)}")
            return None
    
    def _create_scatter_plot(self, df: pd.DataFrame, columns: List[str]) -> Optional[Dict[str, Any]]:
        """Create scatter plot"""
        try:
            if len(columns) < 2:
                return None
            
            fig = go.Figure(data=[
                go.Scatter(
                    x=df[columns[0]],
                    y=df[columns[1]],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=df[columns[1]],
                        colorscale='Viridis',
                        showscale=True
                    )
                )
            ])
            
            fig.update_layout(
                title=f"{columns[0]} vs {columns[1]}",
                xaxis_title=columns[0],
                yaxis_title=columns[1],
                width=700,
                height=500
            )
            
            return {
                "type": "scatter",
                "title": f"{columns[0]} vs {columns[1]}",
                "data": json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))
            }
        except Exception as e:
            logger.error(f"Error creating scatter plot: {str(e)}")
            return None
    
    async def _generate_summary(
        self,
        query: str,
        analysis_results: Dict[str, Any],
        statistics: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate natural language summary of analysis results"""
        
        prompt = f"""
        Generate a clear, concise summary of the data analysis results in response to this query:
        
        Query: {query}
        
        Analysis Results:
        {json.dumps(analysis_results, indent=2)[:3000]}  # Limit length
        
        Statistics:
        {json.dumps(statistics, indent=2)[:1000] if statistics else "N/A"}
        
        Provide a natural language summary that:
        1. Directly answers the user's question
        2. Highlights key findings
        3. Mentions important statistics or trends
        4. Is easy to understand for non-technical users
        """
        
        summary = await self.llm_provider.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=500
        )
        
        return summary
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to dataframe"""
        for column, condition in filters.items():
            if column not in df.columns:
                continue
            
            if isinstance(condition, dict):
                # Complex condition like {"operator": ">", "value": 100}
                operator = condition.get("operator", "==")
                value = condition.get("value")
                
                if operator == ">":
                    df = df[df[column] > value]
                elif operator == "<":
                    df = df[df[column] < value]
                elif operator == ">=":
                    df = df[df[column] >= value]
                elif operator == "<=":
                    df = df[df[column] <= value]
                elif operator == "==":
                    df = df[df[column] == value]
                elif operator == "!=":
                    df = df[df[column] != value]
                elif operator == "in":
                    df = df[df[column].isin(value)]
                elif operator == "contains":
                    df = df[df[column].str.contains(value, case=False, na=False)]
            else:
                # Simple equality filter
                df = df[df[column] == condition]
        
        return df
    
    def _generate_statistics(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate comprehensive statistics for the data"""
        stats = {
            "shape": df.shape,
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
            "column_types": df.dtypes.value_counts().to_dict()
        }
        
        # Numeric statistics
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            stats["numeric_summary"] = {
                "columns": numeric_df.columns.tolist(),
                "correlations": numeric_df.corr().values.tolist() if len(numeric_df.columns) > 1 else None,
                "summary": numeric_df.describe().to_dict()
            }
        
        # Categorical statistics
        categorical_df = df.select_dtypes(include=['object', 'category'])
        if not categorical_df.empty:
            stats["categorical_summary"] = {
                "columns": categorical_df.columns.tolist(),
                "unique_counts": {col: df[col].nunique() for col in categorical_df.columns}
            }
        
        return stats
    
    async def export_results(
        self,
        results: AnalyticsResponse,
        format: str = "json"
    ) -> Union[str, bytes]:
        """Export analysis results in various formats"""
        
        if format == "json":
            return json.dumps({
                "analysis_results": results.analysis_results,
                "summary": results.natural_language_summary,
                "statistics": results.statistics
            }, indent=2)
        
        elif format == "csv":
            # Export data preview as CSV
            df = pd.DataFrame(results.data_preview.get("head", []))
            return df.to_csv(index=False)
        
        elif format == "html":
            # Create HTML report
            html = f"""
            <html>
            <head><title>Data Analysis Report</title></head>
            <body>
                <h1>Analysis Report</h1>
                <h2>Summary</h2>
                <p>{results.natural_language_summary}</p>
                <h2>Results</h2>
                <pre>{json.dumps(results.analysis_results, indent=2)}</pre>
                <h2>Statistics</h2>
                <pre>{json.dumps(results.statistics, indent=2)}</pre>
            </body>
            </html>
            """
            return html
        
        else:
            raise ValueError(f"Unsupported export format: {format}")