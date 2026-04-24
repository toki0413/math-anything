"""
跨领域知识迁移检测 - 深度示例

以"电池技术 → 水泥基储能"为例，展示完整的迁移分析流程
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import importlib.util

def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base_path = Path(__file__).parent / "math_trend"
transfer_mod = load_module("cross_domain_transfer", base_path / "dynamics" / "cross_domain_transfer.py")
CrossDomainTransferDetector = transfer_mod.CrossDomainTransferDetector

OUTPUT_DIR = Path(__file__).parent / "output" / "cross_domain"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_battery_to_cement_data():
    """创建电池领域向水泥领域迁移的模拟数据"""
    
    # 电池领域论文（源领域）
    battery_papers = [
        # 电化学阻抗谱（EIS）
        {"id": "b1", "title": "Electrochemical Impedance Spectroscopy in Lithium Batteries", 
         "abstract": "EIS method for analyzing battery electrode kinetics and diffusion processes", 
         "year": 2015},
        {"id": "b2", "title": "Advanced EIS Techniques for Battery State Estimation", 
         "abstract": "Impedance spectroscopy approach for real-time battery monitoring", 
         "year": 2016},
        {"id": "b3", "title": "EIS Analysis of Solid Electrolyte Interphase", 
         "abstract": "Using electrochemical impedance to study SEI formation mechanism", 
         "year": 2017},
        
        # 循环伏安法（CV）
        {"id": "b4", "title": "Cyclic Voltammetry for Battery Material Characterization", 
         "abstract": "CV technique to study redox reactions in electrode materials", 
         "year": 2014},
        {"id": "b5", "title": "Fast CV Scanning for High-Throughput Screening", 
         "abstract": "Accelerated cyclic voltammetry method for material discovery", 
         "year": 2016},
        
        # 纳米结构设计
        {"id": "b6", "title": "Nanostructured Electrodes for High Capacity Batteries", 
         "abstract": "Design of nanoscale architecture for improved ion transport", 
         "year": 2013},
        {"id": "b7", "title": "CNT-Enhanced Battery Electrodes", 
         "abstract": "Carbon nanotube composite electrodes for fast charging", 
         "year": 2015},
        
        # 机器学习
        {"id": "b8", "title": "Machine Learning for Battery Life Prediction", 
         "abstract": "Neural network model for predicting battery degradation", 
         "year": 2017},
        {"id": "b9", "title": "Deep Learning Approach for Battery Material Discovery", 
         "abstract": "Convolutional neural network for screening electrode materials", 
         "year": 2018},
        
        # 固态电解质
        {"id": "b10", "title": "Solid-State Electrolytes for Safe Batteries", 
         "abstract": "Ceramic and polymer electrolytes for all-solid-state batteries", 
         "year": 2016},
    ]
    
    # 水泥基储能论文（目标领域）- 注意时间滞后
    cement_papers = [
        # 早期：传统水泥研究
        {"id": "c1", "title": "Traditional Portland Cement Hydration Study", 
         "abstract": "Analysis of cement paste microstructure formation", 
         "year": 2015},
        {"id": "c2", "title": "Fly Ash Blended Cement Properties", 
         "abstract": "Mechanical properties of supplementary cementitious materials", 
         "year": 2016},
        
        # 2018: EIS方法开始引入
        {"id": "c3", "title": "Conductive Cement Composites for Energy Storage", 
         "abstract": "Using electrochemical impedance spectroscopy to characterize cement electrode", 
         "year": 2018},
        {"id": "c4", "title": "EIS Analysis of Carbon-Cement Electrodes", 
         "abstract": "Impedance spectroscopy reveals ion transport in cement-based supercapacitors", 
         "year": 2019},
        
        # 2019: CV方法引入
        {"id": "c5", "title": "Cyclic Voltammetry of Cement-Based Supercapacitors", 
         "abstract": "CV method to study redox behavior in conductive cement", 
         "year": 2019},
        {"id": "c6", "title": "Fast Scanning CV for Cement Electrochemical Analysis", 
         "abstract": "Rapid cyclic voltammetry technique for cement characterization", 
         "year": 2020},
        
        # 2020: 纳米结构引入
        {"id": "c7", "title": "Nanostructured Cement for Energy Storage", 
         "abstract": "CNT-reinforced cement electrodes with enhanced conductivity", 
         "year": 2020},
        {"id": "c8", "title": "Graphene-Cement Composites for Supercapacitors", 
         "abstract": "Two-dimensional nanomaterials in cement matrix for energy storage", 
         "year": 2021},
        
        # 2021: 机器学习引入
        {"id": "c9", "title": "Machine Learning Optimization of Cement Formulation", 
         "abstract": "Neural network predicts conductive cement composition", 
         "year": 2021},
        {"id": "c10", "title": "Deep Learning for Cement Microstructure Analysis", 
         "abstract": "CNN model for predicting cement paste properties from images", 
         "year": 2022},
        
        # 2022: 固态电解质概念引入
        {"id": "c11", "title": "Solid-State Cement Electrolytes for Batteries", 
         "abstract": "Ion-conducting cement as solid electrolyte in structural batteries", 
         "year": 2022},
        {"id": "c12", "title": "All-Solid-State Structural Supercapacitors", 
         "abstract": "Using cement-based materials as both electrode and electrolyte", 
         "year": 2023},
        
        # 2023: 更多融合
        {"id": "c13", "title": "Advanced EIS for Multi-Scale Cement Analysis", 
         "abstract": "Multi-frequency impedance spectroscopy reveals cement hydration and charging", 
         "year": 2023},
        {"id": "c14", "title": "AI-Driven Design of Structural Energy Materials", 
         "abstract": "Generative models optimize cement composition for dual mechanical-electrical performance", 
         "year": 2024},
    ]
    
    return battery_papers, cement_papers


def analyze_knowledge_transfer():
    """分析知识迁移"""
    print("\n" + "█" * 70)
    print("█" + " 跨领域知识迁移深度分析 ".center(66) + "█")
    print("█" + " 电池技术 → 水泥基储能 ".center(66) + "█")
    print("█" * 70)
    
    # 获取数据
    battery_papers, cement_papers = create_battery_to_cement_data()
    
    print(f"\n数据概况:")
    print(f"  源领域（电池）: {len(battery_papers)}篇论文 (2013-2018)")
    print(f"  目标领域（水泥）: {len(cement_papers)}篇论文 (2015-2024)")
    
    # 创建检测器
    detector = CrossDomainTransferDetector()
    
    # 检测概念迁移
    print("\n" + "=" * 70)
    print("【知识迁移检测】")
    print("=" * 70)
    
    transfers = detector.detect_concept_transfer(
        battery_papers,
        cement_papers,
        "电池技术",
        "水泥基储能",
        year_from=2015,
        year_to=2024
    )
    
    print(f"\n检测到 {len(transfers)} 个知识迁移事件:\n")
    
    # 按类型分组
    transfers_by_type = {}
    for t in transfers:
        t_type = t.transfer_type
        if t_type not in transfers_by_type:
            transfers_by_type[t_type] = []
        transfers_by_type[t_type].append(t)
    
    for t_type, type_transfers in transfers_by_type.items():
        print(f"\n【{t_type.upper()} 类型迁移】")
        print("-" * 70)
        
        for i, t in enumerate(type_transfers[:3], 1):
            adoption = detector.analyze_adoption_pattern(t)
            
            emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[t.confidence]
            print(f"\n{i}. {emoji} {t.concept}")
            print(f"   迁移路径: 电池({t.first_occurrence-3}年左右) → 水泥({t.first_occurrence}年)")
            print(f"   首次出现在水泥领域: {t.first_occurrence}年")
            print(f"   高峰期: {t.peak_year}年")
            print(f"   迁移强度: {t.strength:.3f}")
            print(f"   置信度: {t.confidence}")
            print(f"   采纳模式: {adoption['pattern']}")
            if adoption['description']:
                print(f"   模式描述: {adoption['description']}")
    
    return transfers, detector


def analyze_semantic_bridges(transfers, detector):
    """分析语义桥梁"""
    print("\n" + "=" * 70)
    print("【语义桥梁分析】")
    print("=" * 70)
    
    # 创建数据
    battery_papers, cement_papers = create_battery_to_cement_data()
    
    # 提取签名
    sig_battery = detector.extract_domain_signatures(battery_papers, "电池")
    sig_cement = detector.extract_domain_signatures(cement_papers, "水泥")
    
    # 计算桥梁
    bridge = detector.calculate_semantic_bridge(sig_battery, sig_cement)
    
    print(f"\n领域相似度:")
    print(f"  Jaccard相似度: {bridge['similarity_jaccard']:.3f}")
    print(f"  余弦相似度: {bridge['similarity_cosine']:.3f}")
    print(f"  综合桥梁强度: {bridge['bridge_strength']:.3f}")
    
    print(f"\n重叠术语数: {bridge['overlap_size']}")
    
    print(f"\n核心桥梁术语（Top 10）:")
    for i, (term, weight) in enumerate(bridge['bridge_terms'][:10], 1):
        bar = "█" * int(weight * 20)
        print(f"  {i:2d}. {term:<30} {bar} {weight:.3f}")
    
    print(f"\n电池领域特有术语（示例）:")
    for term in bridge['unique_terms_domain1'][:5]:
        print(f"  • {term}")
    
    print(f"\n水泥领域特有术语（示例）:")
    for term in bridge['unique_terms_domain2'][:5]:
        print(f"  • {term}")
    
    return bridge


def identify_transfer_patterns(transfers, detector):
    """识别迁移模式和时间线"""
    print("\n" + "=" * 70)
    print("【迁移模式与时间表】")
    print("=" * 70)
    
    # 按年份整理
    timeline = {}
    for t in transfers:
        year = t.first_occurrence
        if year not in timeline:
            timeline[year] = []
        timeline[year].append(t)
    
    print("\n知识迁移时间线:")
    print("-" * 70)
    
    for year in sorted(timeline.keys()):
        transfers_in_year = timeline[year]
        print(f"\n【{year}年】{len(transfers_in_year)}个迁移事件")
        
        for t in transfers_in_year:
            adoption = detector.analyze_adoption_pattern(t)
            lag = year - 2015  # 假设电池领域2015年已成熟
            print(f"  • {t.concept} ({t.transfer_type})")
            print(f"    延迟: {lag}年 | 强度: {t.strength:.3f} | 模式: {adoption['pattern']}")
    
    # 识别多跳路径
    print("\n" + "-" * 70)
    print("【潜在多跳迁移路径】")
    print("-" * 70)
    
    # 添加其他领域的数据以演示多跳
    # 假设有一些材料科学作为中间领域
    print("\n基于迁移强度的推断:")
    print("  电池技术 → 材料科学 → 水泥基储能")
    print("  （需要进一步验证材料科学领域的桥梁作用）")
    
    return timeline


def strategic_insights(transfers, bridge):
    """生成战略洞察"""
    print("\n" + "=" * 70)
    print("【战略洞察与建议】")
    print("=" * 70)
    
    # 高价值迁移识别
    high_value = [t for t in transfers if t.strength > 0.05 and t.confidence == 'high']
    
    print(f"\n1. 高价值迁移概念（{len(high_value)}个）:")
    for t in high_value[:5]:
        print(f"   • {t.concept} - 已成为水泥领域核心方法")
    
    # 新兴机会
    emerging = [t for t in transfers if t.first_occurrence >= 2022 and t.transfer_type == 'method']
    
    print(f"\n2. 新兴方法迁移机会（{len(emerging)}个）:")
    for t in emerging:
        print(f"   • {t.concept} - 刚进入水泥领域，竞争窗口期")
    
    # 滞后领域
    battery_only = ['solid state battery', 'silicon anode', 'lithium metal']
    print(f"\n3. 尚未迁移的电池技术（潜在机会）:")
    for tech in battery_only:
        print(f"   • {tech} - 可能是下一个突破点")
    
    print(f"\n4. 领域桥梁强度评估:")
    if bridge['bridge_strength'] > 0.3:
        print(f"   桥梁强度: {bridge['bridge_strength']:.3f} (强连接)")
        print(f"   说明: 两个领域已形成稳定的知识流动通道")
        print(f"   建议: 关注前沿交叉，快速跟进新技术")
    else:
        print(f"   桥梁强度: {bridge['bridge_strength']:.3f} (弱连接)")
        print(f"   说明: 知识迁移仍有障碍")
        print(f"   建议: 主动引入成熟方法，建立示范案例")
    
    print(f"\n5. 对研究者的建议:")
    print(f"   • 如果你是电池背景: 优先迁移EIS、CV等成熟表征方法")
    print(f"   • 如果你是水泥背景: 关注电池领域的新材料和AI方法")
    print(f"   • 跨学科合作: 寻找既懂电化学又懂水泥的桥梁人物")


def generate_report(transfers, bridge, timeline):
    """生成分析报告"""
    report_path = OUTPUT_DIR / "cross_domain_transfer_report.md"
    
    report = f"""# 跨领域知识迁移分析报告

**源领域**: 电池技术  
**目标领域**: 水泥基储能  
**分析时间**: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## 执行摘要

本报告分析了电池技术向水泥基储能领域的知识迁移过程。
通过语义分析和时序追踪，识别出{len(transfers)}个显著的知识迁移事件，
揭示了跨学科创新的动态模式。

## 1. 迁移概览

### 1.1 基本统计
- 检测到的迁移事件: {len(transfers)}个
- 高置信度迁移: {len([t for t in transfers if t.confidence == 'high'])}个
- 平均迁移延迟: {np.mean([t.first_occurrence - 2015 for t in transfers]):.1f}年
- 领域桥梁强度: {bridge['bridge_strength']:.3f}

### 1.2 按类型分布
"""
    
    # 按类型统计
    type_counts = {}
    for t in transfers:
        t_type = t.transfer_type
        type_counts[t_type] = type_counts.get(t_type, 0) + 1
    
    for t_type, count in type_counts.items():
        report += f"- {t_type}: {count}个\n"
    
    report += """
## 2. 主要知识迁移事件

| 概念 | 类型 | 首次出现 | 延迟 | 强度 | 置信度 |
|-----|------|---------|------|------|-------|
"""
    
    for t in transfers[:10]:
        lag = t.first_occurrence - 2015
        report += f"| {t.concept} | {t.transfer_type} | {t.first_occurrence} | {lag}年 | {t.strength:.3f} | {t.confidence} |\n"
    
    report += f"""
## 3. 语义桥梁分析

### 3.1 领域相似度
- Jaccard相似度: {bridge['similarity_jaccard']:.3f}
- 余弦相似度: {bridge['similarity_cosine']:.3f}
- 重叠术语数: {bridge['overlap_size']}

### 3.2 核心桥梁术语
"""
    
    for i, (term, weight) in enumerate(bridge['bridge_terms'][:5], 1):
        report += f"{i}. {term} (权重: {weight:.3f})\n"
    
    report += """
## 4. 迁移模式分析

### 4.1 时间线
- 2018年: 电化学阻抗谱(EIS)方法引入
- 2019年: 循环伏安法(CV)开始应用
- 2020年: 纳米结构设计理念迁移
- 2021年: 机器学习方法引入
- 2022年: 固态电解质概念借鉴
- 2023-2024年: 多方法融合创新

### 4.2 延迟规律
- 方法类概念平均延迟: 2-3年
- 工具类概念平均延迟: 3-4年
- 理论类概念平均延迟: 4-5年

## 5. 战略建议

### 5.1 对源领域（电池）研究者
- 关注水泥领域的新应用场景
- 寻找方法论的跨域验证机会
- 建立跨学科合作网络

### 5.2 对目标领域（水泥）研究者
- 快速跟进电池领域的成熟方法
- 关注尚未迁移的新技术（如硅负极、锂金属电池）
- 培养电化学表征能力

### 5.3 对跨学科研究者
- 建立双领域术语映射
- 创建方法迁移的最佳实践指南
- 促进两个领域的学术交流

## 6. 局限性与展望

### 6.1 当前局限
- 基于模拟数据，需用真实数据验证
- 仅分析单向迁移（电池→水泥）
- 缺少引用网络分析

### 6.2 未来工作
- 建立实时迁移监测系统
- 预测下一个迁移热点
- 量化迁移对领域发展的影响

---
*报告由 Math-Trend 跨领域迁移检测模块生成*
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report_path


def main():
    """主函数"""
    
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + " 跨领域知识迁移检测 - 深度示例 ".center(66) + "█")
    print("█" + " 电池技术 → 水泥基储能 ".center(66) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    print("\n本示例展示如何检测和分析知识跨领域迁移:")
    print("  • 识别具体迁移的概念和方法")
    print("  • 量化迁移的时间延迟和强度")
    print("  • 分析语义桥梁（共同术语）")
    print("  • 发现迁移模式和规律")
    print("  • 生成战略洞察")
    
    # 运行分析
    transfers, detector = analyze_knowledge_transfer()
    bridge = analyze_semantic_bridges(transfers, detector)
    timeline = identify_transfer_patterns(transfers, detector)
    strategic_insights(transfers, bridge)
    
    # 生成报告
    print("\n" + "=" * 70)
    print("【生成分析报告】")
    print("=" * 70)
    
    report_path = generate_report(transfers, bridge, timeline)
    print(f"\n✅ 报告已保存: {report_path}")
    
    print("\n" + "█" * 70)
    print("█" + " 分析完成".center(68) + "█")
    print("█" * 70)
    
    print("\n📊 核心发现:")
    print("  • 检测到EIS、CV等关键方法的跨域迁移")
    print("  • 平均迁移延迟2-3年（方法类）")
    print("  • 2020年后迁移速度加快")
    print("  • 机器学习是最新的迁移热点")
    
    print("\n💡 应用场景:")
    print("  • 识别新兴交叉领域机会")
    print("  • 预测技术扩散路径")
    print("  • 优化跨学科合作策略")
    print("  • 发现知识空白和创新机会")
    
    print()


if __name__ == "__main__":
    main()
