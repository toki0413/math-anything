"""
时间序列分析模块 - 插件化实现

功能：
- 年度趋势分析
- 拐点检测
- 简单预测
"""

from collections import Counter
from datetime import datetime
from typing import List

import numpy as np

from core.data_model import Paper, AnalysisResult
from core.config import AnalysisConfig
from core.interfaces import BaseAnalyzer


class TimeSeriesAnalyzer(BaseAnalyzer):
    """时间序列分析"""
    
    def name(self) -> str:
        return "advanced.time_series"
    
    def analyze(self, papers: List[Paper]) -> AnalysisResult:
        yearly = Counter(p.year for p in papers if p.year > 0)
        years = sorted(yearly.keys())
        counts = [yearly[y] for y in years]
        
        if len(years) < 3:
            return AnalysisResult(
                module_name=self.name(),
                timestamp=datetime.now(),
                success=False,
                data={},
                error=f"数据点不足（{len(years)}个，需要≥3个）"
            )
        
        # 增长率
        growth_rates = []
        for i in range(1, len(counts)):
            if counts[i-1] > 0:
                growth_rates.append((counts[i] - counts[i-1]) / counts[i-1])
        
        avg_growth = np.mean(growth_rates) if growth_rates else 0
        
        # CAGR
        first_count = counts[0]
        last_count = counts[-1]
        n_years = years[-1] - years[0]
        cagr = ((last_count / first_count) ** (1 / n_years) - 1) * 100 if first_count > 0 and n_years > 0 else 0
        
        # 拐点检测（二阶差分）
        inflection_points = []
        if len(counts) >= 3:
            for i in range(1, len(counts) - 1):
                accel = counts[i+1] - 2*counts[i] + counts[i-1]
                if abs(accel) > np.std(counts) * 0.5:
                    inflection_points.append({
                        "year": years[i],
                        "count": counts[i],
                        "acceleration": accel,
                        "type": "加速" if accel > 0 else "减速"
                    })
        
        # 简单线性预测
        if len(years) >= 3:
            x = np.array(years)
            y = np.array(counts)
            coeffs = np.polyfit(x, y, 1)
            predicted = {}
            for i in range(1, 4):
                future_year = years[-1] + i
                pred = max(0, int(np.polyval(coeffs, future_year)))
                predicted[future_year] = pred
        else:
            predicted = {}
        
        result_data = {
            "yearly_data": dict(yearly),
            "growth_rates": {str(y): round(r, 3) for y, r in zip(years[1:], growth_rates)},
            "avg_growth_rate": round(avg_growth * 100, 1),
            "cagr_percent": round(cagr, 1),
            "inflection_points": inflection_points,
            "predictions": predicted,
        }
        
        return AnalysisResult(
            module_name=self.name(),
            timestamp=datetime.now(),
            success=True,
            data=result_data
        )


def get_analyzer(config: AnalysisConfig) -> BaseAnalyzer:
    return TimeSeriesAnalyzer(config)
