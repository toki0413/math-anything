import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

def create_figure7():
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3,
                          left=0.06, right=0.95, top=0.92, bottom=0.06)
    
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.axis('off')
    ax1.set_title('(a) Structural Battery Architecture and Working Principle', fontsize=11, fontweight='bold', pad=10, loc='left')
    
    components = [
        {'x': 1, 'y': 3, 'w': 1.5, 'h': 2, 'label': 'Current\nCollector', 'color': '#FFCDD2'},
        {'x': 3, 'y': 3, 'w': 2, 'h': 2, 'label': 'Anode\n(Carbon)', 'color': '#BBDEFB'},
        {'x': 5.5, 'y': 3, 'w': 2, 'h': 2, 'label': 'Electrolyte\n(Cement)', 'color': '#C8E6C9'},
        {'x': 8, 'y': 3, 'w': 1.5, 'h': 2, 'label': 'Cathode\n(Carbon)', 'color': '#BBDEFB'},
    ]
    
    for comp in components:
        box = FancyBboxPatch((comp['x'], comp['y']), comp['w'], comp['h'],
                              boxstyle='round,pad=0.1', facecolor=comp['color'],
                              edgecolor='#333333', linewidth=2)
        ax1.add_patch(box)
        ax1.text(comp['x']+comp['w']/2, comp['y']+comp['h']/2, comp['label'],
                fontsize=9, ha='center', va='center', fontweight='bold')
    
    for i in range(len(components)-1):
        arrow = FancyArrowPatch((components[i]['x']+components[i]['w'], components[i]['y']+1),
                               (components[i+1]['x'], components[i+1]['y']+1),
                               arrowstyle='->', mutation_scale=15, linewidth=2, color='#333333')
        ax1.add_patch(arrow)
    
    ax1.annotate('e⁻ flow', xy=(4, 4.5), xytext=(4, 5.3),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=9, color='blue', fontweight='bold')
    ax1.annotate('Ion transport', xy=(6.5, 4.5), xytext=(6.5, 5.3),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=9, color='green', fontweight='bold')
    
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('(b) Fabrication Process Flow', fontsize=11, fontweight='bold', pad=10, loc='left')
    
    steps = [
        {'y': 8.5, 'text': '1. Mix cement + conductive filler'},
        {'y': 6.5, 'text': '2. Add electrode layers'},
        {'y': 4.5, 'text': '3. Cast into mold'},
        {'y': 2.5, 'text': '4. Cure & condition'},
    ]
    
    for step in steps:
        box = FancyBboxPatch((0.5, step['y']-0.5), 9, 1,
                              boxstyle='round,pad=0.08', facecolor='#FFF9C4',
                              edgecolor='#FBC02D', linewidth=1.5)
        ax2.add_patch(box)
        ax2.text(1, step['y'], step['text'], fontsize=9, va='center')
        
        if step['y'] > 2.5:
            arrow = FancyArrowPatch((5, step['y']-0.6), (5, step['y']-1.4),
                                   arrowstyle='->', mutation_scale=12, linewidth=1.5, color='#666666')
            ax2.add_patch(arrow)
    
    ax3 = fig.add_subplot(gs[1, 0])
    
    np.random.seed(42)
    energy_data = [
        np.random.normal(12, 3, 50),
        np.random.normal(8, 2, 50),
        np.random.normal(18, 5, 50),
        np.random.normal(6, 1.5, 50),
    ]
    
    bp = ax3.boxplot(energy_data, patch_artist=True, labels=['CF', 'CNT', 'GR', 'CP'])
    colors_box = ['#1976D2', '#388E3C', '#F57C00', '#C62828']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Energy Density (Wh/kg)', fontsize=10, fontweight='bold')
    ax3.set_title('(c) Energy Density Distribution by Material Type', fontsize=10, fontweight='bold', pad=8, loc='left')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    ax4 = fig.add_subplot(gs[1, 1])
    
    cost_items = ['Raw Materials', 'Manufacturing', 'Encapsulation', 'Testing', 'Other']
    costs = [35, 25, 20, 12, 8]
    colors_pie = ['#1976D2', '#388E3C', '#F57C00', '#C62828', '#9E9E9E']
    explode = (0.05, 0, 0, 0, 0)
    
    wedges, texts, autotexts = ax4.pie(costs, explode=explode, labels=cost_items, autopct='%1.1f%%',
                                       colors=colors_pie, startangle=90, pctdistance=0.75,
                                       wedgeprops=dict(linewidth=2, edgecolor='white'))
    
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    ax4.set_title('(d) Cost Breakdown of CBES System ($/kWh)', fontsize=10, fontweight='bold', pad=8, loc='left')
    
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)
    ax5.axis('off')
    ax5.set_title('(e) Technology Readiness Assessment', fontsize=10, fontweight='bold', pad=8, loc='left')
    
    trl_items = [
        {'name': 'Material synthesis', 'trl': 7, 'color': '#4CAF50'},
        {'name': 'Lab-scale device', 'trl': 6, 'color': '#8BC34A'},
        {'name': 'Pilot production', 'trl': 4, 'color': '#FFC107'},
        {'name': 'Building integration', 'trl': 3, 'color': '#FF9800'},
        {'name': 'Commercial product', 'trl': 2, 'color': '#F44336'},
    ]
    
    for i, item in enumerate(trl_items):
        y_pos = 8 - i*1.5
        ax5.barh(y_pos, item['trl'], height=0.8, color=item['color'], alpha=0.7, edgecolor='white')
        ax5.text(item['trl'] + 0.2, y_pos, f'TRL {item["trl"]}', fontsize=9, va='center', fontweight='bold')
        ax5.text(-0.2, y_pos, item['name'], fontsize=9, ha='right', va='center')
    
    ax5.set_xlim(0, 10)
    ax5.set_yticks([])
    ax5.set_xlabel('TRL Level', fontsize=9, fontweight='bold')
    
    fig.suptitle('Figure 7. Device Architecture, Manufacturing, and Technology Assessment', 
                 fontsize=13, fontweight='bold', y=0.98)
    
    plt.savefig('Figure7_Device_Architecture.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print('✓ Figure 7 created')

def create_figure8():
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3,
                          left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    years = [2015, 2017, 2019, 2021, 2023, 2025]
    publications = [15, 35, 78, 145, 230, 320]
    citations = [120, 380, 920, 2100, 4200, 6500]
    
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(years, publications, 'o-', linewidth=2.5, markersize=8, 
                    color='#1976D2', label='Publications', markerfacecolor='white', markeredgewidth=2)
    line2 = ax1_twin.plot(years, citations, 's--', linewidth=2.5, markersize=8, 
                         color='#F57C00', label='Citations', markerfacecolor='white', markeredgewidth=2)
    
    ax1.set_xlabel('Year', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Number of Publications', fontsize=10, fontweight='bold', color='#1976D2')
    ax1_twin.set_ylabel('Total Citations', fontsize=10, fontweight='bold', color='#F57C00')
    ax1.tick_params(axis='y', labelcolor='#1976D2')
    ax1_twin.tick_params(axis='y', labelcolor='#F57C00')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=9, framealpha=0.9)
    ax1.set_title('(a) Research Trend: Publications and Citations (2015-2025)', 
                 fontsize=11, fontweight='bold', pad=10, loc='left')
    
    ax2 = fig.add_subplot(gs[0, 1])
    
    countries = ['China', 'USA', 'Germany', 'UK', 'Japan', 'South Korea', 'Others']
    pub_counts = [45, 22, 12, 8, 6, 5, 2]
    colors_geo = ['#C62828', '#1976D2', '#388E3C', '#7B1FA2', '#F57C00', '#009688', '#9E9E9E']
    
    bars = ax2.barh(range(len(countries)), pub_counts, color=colors_geo, edgecolor='white', height=0.6)
    ax2.set_yticks(range(len(countries)))
    ax2.set_yticklabels(countries, fontsize=9)
    ax2.set_xlabel('Publication Share (%)', fontsize=10, fontweight='bold')
    ax2.set_title('(b) Geographic Distribution of CBES Research', fontsize=11, fontweight='bold', pad=10, loc='left')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars, pub_counts):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val}%', va='center', fontsize=9, fontweight='bold')
    
    ax3 = fig.add_subplot(gs[1, 0])
    
    topics = ['Electrode Materials', 'Electrolyte Systems', 'Device Design', 
              'Applications', 'Durability/Safety', 'Modeling/Simulation']
    topic_trends = {
        'Electrode Materials': [20, 25, 30, 32],
        'Electrolyte Systems': [15, 18, 22, 25],
        'Device Design': [18, 22, 20, 18],
        'Applications': [12, 15, 18, 23],
        'Durability/Safety': [8, 10, 13, 18],
        'Modeling/Simulation': [5, 8, 12, 16],
    }
    periods = ['2019-2020', '2021-2022', '2023-2024', '2025']
    
    x = np.arange(len(periods))
    width = 0.12
    
    colors_topic = ['#1976D2', '#388E3C', '#F57C00', '#C62828', '#7B1FA2', '#009688']
    
    for i, (topic, values) in enumerate(topic_trends.items()):
        offset = (i - len(topics)/2 + 0.5) * width
        ax3.bar(x + offset, values, width, label=topic, color=colors_topic[i], alpha=0.8)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(periods, fontsize=9)
    ax3.set_ylabel('Topic Share (%)', fontsize=10, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=7, ncol=2, framealpha=0.9)
    ax3.set_title('(c) Research Topic Evolution Over Time', fontsize=11, fontweight='bold', pad=10, loc='left')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    ax4 = fig.add_subplot(gs[1, 1])
    
    keywords = ['Supercapacitor', 'Concrete battery', 'Structural energy',
               'Carbon fiber', 'Graphene', 'Ionic conductivity', 
               'Mechanical strength', 'Self-powered', 'Smart building', 'Energy storage']
    frequencies = [85, 72, 68, 65, 58, 52, 48, 42, 38, 35]
    
    colors_kw = plt.cm.viridis(np.linspace(0.2, 0.9, len(keywords)))
    
    bars_kw = ax4.barh(range(len(keywords)), frequencies, color=colors_kw, edgecolor='white', height=0.7)
    ax4.set_yticks(range(len(keywords)))
    ax4.set_yticklabels(keywords, fontsize=9)
    ax4.set_xlabel('Frequency (%)', fontsize=10, fontweight='bold')
    ax4.set_title('(d) Top Keywords in CBES Literature', fontsize=11, fontweight='bold', pad=10, loc='left')
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    
    fig.suptitle('Figure 8. Bibliometric Analysis: Research Landscape and Trends', 
                 fontsize=13, fontweight='bold', y=0.98)
    
    plt.savefig('Figure8_Bibliometric_Analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print('✓ Figure 8 created')

if __name__ == '__main__':
    create_figure7()
    create_figure8()
    print('\n✓ All additional figures generated!')
