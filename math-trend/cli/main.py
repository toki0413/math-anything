"""
Math-Trend 统一CLI入口

新架构下的唯一入口：
  math-trend run              运行完整Pipeline
  math-trend run --module basic.statistics  运行单个模块
  math-trend report           生成报告
  math-trend config           查看/编辑配置
"""

import sys
import argparse
from pathlib import Path

# 确保项目根目录在sys.path中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import AnalysisConfig
from core.pipeline import PipelineEngine
from reporting.renderer import ReportRenderer


def cmd_run(args):
    """运行分析Pipeline"""
    config = AnalysisConfig()
    
    if args.domain:
        config.domain_name = args.domain
    if args.year_from:
        config.year_from = args.year_from
    if args.year_to:
        config.year_to = args.year_to
    if args.modules:
        config.enabled_modules = args.modules
    
    engine = PipelineEngine(config)
    results = engine.run(force_refresh=args.refresh)
    
    # 保存结果
    output_path = engine.save_results()
    
    # 生成报告
    if args.report:
        renderer = ReportRenderer(template=config.report_template)
        for name, result in results.items():
            renderer.add_result(result)
        report_path = PROJECT_ROOT / config.output_dir / "report.md"
        renderer.save(report_path, config)
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("【运行完成】")
    print("=" * 70)
    success = sum(1 for r in results.values() if r.success)
    print(f"成功: {success}/{len(results)}个模块")


def cmd_report(args):
    """仅生成报告（从已有结果）"""
    results_path = PROJECT_ROOT / "output" / "pipeline_results.json"
    
    if not results_path.exists():
        print(f"❌ 结果文件不存在: {results_path}")
        print("   请先运行: math-trend run")
        return
    
    import json
    with open(results_path, "r", encoding="utf-8") as f:
        results_data = json.load(f)
    
    renderer = ReportRenderer(template="data_driven")
    
    from core.data_model import AnalysisResult
    from datetime import datetime
    
    for name, data in results_data.items():
        result = AnalysisResult(
            module_name=name,
            timestamp=datetime.now(),
            success=data.get("success", False),
            data=data.get("data", {}),
            error=data.get("error"),
        )
        renderer.add_result(result)
    
    report_path = PROJECT_ROOT / "output" / "report.md"
    renderer.save(report_path)


def main():
    parser = argparse.ArgumentParser(
        description="Math-Trend 研究趋势分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  math-trend run                          运行完整分析
  math-trend run --domain "钙钛矿光伏"     切换研究领域
  math-trend run --modules basic.statistics  只运行统计模块
  math-trend report                       从已有结果生成报告
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # run 命令
    run_parser = subparsers.add_parser("run", help="运行分析Pipeline")
    run_parser.add_argument("--domain", help="研究领域名称")
    run_parser.add_argument("--year-from", type=int, dest="year_from")
    run_parser.add_argument("--year-to", type=int, dest="year_to")
    run_parser.add_argument("--modules", nargs="*", help="指定要运行的模块")
    run_parser.add_argument("--report", action="store_true", help="运行后生成报告")
    run_parser.add_argument("--refresh", action="store_true", help="强制刷新数据（忽略缓存）")
    run_parser.set_defaults(func=cmd_run)
    
    # report 命令
    report_parser = subparsers.add_parser("report", help="生成报告")
    report_parser.set_defaults(func=cmd_report)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
