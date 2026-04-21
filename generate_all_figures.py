import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Wedge
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

def create_figure5():
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3,
                          left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    materials = {
        'Carbon Fiber': {'energy': [8, 12, 15], 'power': [50, 80, 120], 'color': '#1976D2'},
        'CNT-Cement': {'energy': [6, 9, 12], 'power': [100, 150, 200], 'color': '#388E3C'},
        'Graphene Aerogel': {'energy': [15, 20, 28], 'power': [30, 45, 60], 'color': '#F57C00'},
        'Conductive Polymer': {'energy': [4, 7, 10], 'power': [200, 280, 350], 'color': '#C62828'},
        'Activated Carbon': {'energy': [3, 5, 8], 'power': [500, 700, 900], 'color': '#7B1FA2'},
        'Li-ion (ref)': {'energy': [120, 150, 180], 'power': [250, 350, 450], 'color': '#607D8B', 'marker': '*'},
    }
    
    for name, data in materials.items():
        marker = data.get('marker', 'o')
        ax1.scatter(data['power'], data['energy'], 
                   c=data['color'], label=name, s=80, marker=marker,
                   alpha=0.7, edgecolors='white', linewidths=1)
    
    ax1.set_xlabel('Power Density (W/kg)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Energy Density (Wh/kg)', fontsize=10, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax1.set_title('(a) Ragone Plot: Energy vs Power Density', fontsize=11, fontweight='bold', pad=10, loc='left')
    
    ax1.fill_between([10, 100], [1, 1], [10, 10], alpha=0.1, color='red', label='_nolegend_')
    ax1.text(30, 3, 'Supercapacitor\nRegion', fontsize=8, ha='center', color='red', alpha=0.7)
    ax1.fill_between([200, 1000], [50, 50], [200, 200], alpha=0.1, color='blue', label='_nolegend_')
    ax1.text(400, 100, 'Battery\nRegion', fontsize=8, ha='center', color='blue', alpha=0.7)
    
    ax2 = fig.add_subplot(gs[0, 1])
    
    electrodes = ['Carbon\nFiber', 'CNT-\nModified', 'Graphene\nAerogel', 'Conductive\nPolymer', 'Activated\nCarbon']
    capacitances = [85, 120, 180, 65, 220]
    errors = [15, 25, 40, 12, 35]
    colors_bars = ['#1976D2', '#388E3C', '#F57C00', '#C62828', '#7B1FA2']
    
    bars = ax2.bar(range(len(electrodes)), capacitances, yerr=errors, 
                   capsize=5, color=colors_bars, edgecolor='white', linewidth=1.5,
                   error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    
    ax2.set_xticks(range(len(electrodes)))
    ax2.set_xticklabels(electrodes, fontsize=9)
    ax2.set_ylabel('Specific Capacitance (F/g)', fontsize=10, fontweight='bold')
    ax2.set_title('(b) Electrode Material Capacitance Comparison', fontsize=11, fontweight='bold', pad=10, loc='left')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val, err in zip(bars, capacitances, errors):
        ax2.text(bar.get_x() + bar.get_width()/2, val + err + 5,
                str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax3 = fig.add_subplot(gs[1, 0])
    
    cycles = np.array([0, 500, 1000, 2000, 5000, 10000])
    
    cf_retention = np.array([100, 98, 95, 90, 82, 72])
    cnt_retention = np.array([100, 97, 94, 88, 78, 68])
    graphene_retention = np.array([100, 96, 92, 84, 70, 55])
    polymer_retention = np.array([100, 95, 89, 78, 58, 38])
    
    ax3.plot(cycles, cf_retention, 'o-', linewidth=2, markersize=6, color='#1976D2', label='Carbon Fiber')
    ax3.plot(cycles, cnt_retention, 's-', linewidth=2, markersize=6, color='#388E3C', label='CNT-Modified')
    ax3.plot(cycles, graphene_retention, '^-', linewidth=2, markersize=6, color='#F57C00', label='Graphene Aerogel')
    ax3.plot(cycles, polymer_retention, 'd-', linewidth=2, markersize=6, color='#C62828', label='Conductive Polymer')
    
    ax3.axhline(y=80, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='80% threshold')
    ax3.set_xlabel('Cycle Number', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Capacity Retention (%)', fontsize=10, fontweight='bold')
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax3.set_title('(c) Cycling Stability Comparison', fontsize=11, fontweight='bold', pad=10, loc='left')
    ax3.set_xscale('log')
    
    ax4 = fig.add_subplot(gs[1, 1])
    
    conductivity = [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    strength = [45, 42, 38, 32, 26, 18, 12]
    strength_err = [2, 2.5, 3, 3.5, 4, 4.5, 5]
    
    ax4.errorbar(conductivity, strength, yerr=strength_err, fmt='o-', 
                color='#E91E63', linewidth=2, markersize=8, capsize=4,
                markerfacecolor='white', markeredgewidth=2)
    
    ax4.fill_between(conductivity, 
                     [s-e for s,e in zip(strength, strength_err)],
                     [s+e for s,e in zip(strength, strength_err)], 
                     alpha=0.2, color='#E91E63')
    
    ax4.set_xlabel('Electrical Conductivity (S/cm)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Compressive Strength (MPa)', fontsize=10, fontweight='bold')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_title('(d) Mechanical-Electrical Trade-off', fontsize=11, fontweight='bold', pad=10, loc='left')
    
    ax4.annotate('Optimal Region\n(0.1-1 S/cm)', xy=(0.5, 32), xytext=(2, 35),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=9, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='red'))
    
    fig.suptitle('Figure 5. Comprehensive Performance Analysis of CBES Electrode Materials', 
                 fontsize=13, fontweight='bold', y=0.98)
    
    plt.savefig('Figure5_Electrode_Performance.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print('✓ Figure 5 created')

def create_figure6():
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3,
                          left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    ax1 = fig.add_subplot(gs[0, 0], polar=True)
    
    categories = ['Ionic Conduct.', 'Voltage Window', 'Stability', 'Cost', 'Safety', 'Compat.']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    electrolytes = {
        'Aqueous (KOH)': [95, 40, 70, 98, 75, 90],
        'Gel Polymer': [60, 55, 85, 70, 90, 95],
        'Ionic Liquid': [45, 95, 95, 30, 85, 50],
        'Solid State': [30, 80, 98, 50, 99, 70],
    }
    colors_radar = ['#1976D2', '#388E3C', '#F57C00', '#C62828']
    
    for (name, values), color in zip(electrolytes.items(), colors_radar):
        values += values[:1]
        ax1.plot(angles, values, 'o-', linewidth=2, label=name, color=color, markersize=5)
        ax1.fill(angles, values, alpha=0.1, color=color)
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, fontsize=8)
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=8)
    ax1.set_title('(a) Electrolyte System Multi-criteria Comparison', fontsize=11, fontweight='bold', pad=20, loc='left')
    
    ax2 = fig.add_subplot(gs[0, 1])
    
    concentrations = np.linspace(0, 14, 100)
    koh_conductivity = 0.5 * concentrations * np.exp(-concentrations/10) * 100
    nacl_conductivity = 0.4 * concentrations * np.exp(-concentrations/12) * 100
    h2so4_conductivity = 0.8 * concentrations * np.exp(-concentrations/8) * 100
    
    ax2.plot(concentrations, koh_conductivity, '-', linewidth=2.5, color='#1976D2', label='KOH')
    ax2.plot(concentrations, nacl_conductivity, '--', linewidth=2.5, color='#388E3C', label='NaCl')
    ax2.plot(concentrations, h2so4_conductivity, '-.', linewidth=2.5, color='#F57C00', label='H₂SO₄')
    
    ax2.set_xlabel('Concentration (M)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Ionic Conductivity (mS/cm)', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax2.set_title('(b) Ionic Conductivity vs Concentration', fontsize=11, fontweight='bold', pad=10, loc='left')
    
    max_idx_koh = np.argmax(koh_conductivity)
    ax2.scatter([concentrations[max_idx_koh]], [koh_conductivity[max_idx_koh]], 
               s=150, color='red', zorder=5, marker='*')
    ax2.annotate(f'Optimal: {concentrations[max_idx_koh]:.1f} M\n({koh_conductivity[max_idx_koh]:.0f} mS/cm)',
                xy=(concentrations[max_idx_koh], koh_conductivity[max_idx_koh]),
                xytext=(concentrations[max_idx_koh]+2, koh_conductivity[max_idx_koh]+20),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, fontweight='bold', color='red')
    
    ax3 = fig.add_subplot(gs[1, 0])
    
    voltage_range = np.linspace(0, 2.5, 100)
    
    aqueous_stable = np.where((voltage_range >= 0) & (voltage_range <= 1.0), 1, 0)
    gel_stable = np.where((voltage_range >= 0) & (voltage_range <= 1.5), 1, 0)
    il_stable = np.where((voltage_range >= 0) & (voltage_range <= 3.0), 1, 0)
    
    ax3.fill_between(voltage_range, 0, aqueous_stable, alpha=0.3, color='#1976D2', label='Aqueous (≤1.0 V)')
    ax3.fill_between(voltage_range, 0, gel_stable, alpha=0.3, color='#388E3C', label='Gel Polymer (≤1.5 V)')
    ax3.fill_between(voltage_range, 0, il_stable, alpha=0.3, color='#F57C00', label='Ionic Liquid (≤3.0 V)')
    
    ax3.set_xlabel('Applied Voltage (V)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Electrochemical Stability', fontsize=10, fontweight='bold')
    ax3.set_xlim(0, 2.5)
    ax3.set_ylim(0, 1.2)
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax3.set_yticks([0, 0.5, 1])
    ax3.set_yticklabels(['Unstable', '', 'Stable'])
    ax3.set_title('(c) Electrochemical Stability Window', fontsize=11, fontweight='bold', pad=10, loc='left')
    ax3.axvline(x=1.23, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax3.text(1.25, 0.5, 'Water splitting\nlimit (1.23V)', fontsize=8, rotation=90, va='center', color='gray')
    
    ax4 = fig.add_subplot(gs[1, 1])
    
    temperatures = np.array([-20, 0, 20, 40, 60, 80])
    
    cap_ambient = np.array([70, 85, 100, 105, 102, 90])
    cap_freeze_protected = np.array([88, 95, 100, 103, 101, 92])
    cap_high_temp = np.array([75, 88, 100, 108, 115, 110])
    
    ax4.plot(temperatures, cap_ambient, 'o-', linewidth=2, markersize=7, 
            color='#1976D2', label='Standard Cement', markerfacecolor='white', markeredgewidth=1.5)
    ax4.plot(temperatures, cap_freeze_protected, 's-', linewidth=2, markersize=7, 
            color='#388E3C', label='Freezing-resistant', markerfacecolor='white', markeredgewidth=1.5)
    ax4.plot(temperatures, cap_high_temp, '^-', linewidth=2, markersize=7, 
            color='#F57C00', label='High-temp optimized', markerfacecolor='white', markeredgewidth=1.5)
    
    ax4.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax4.text(-18, 101, 'Baseline (25°C)', fontsize=8, color='gray')
    
    ax4.set_xlabel('Temperature (°C)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Relative Capacity (%)', fontsize=10, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax4.set_title('(d) Temperature Dependence of Performance', fontsize=11, fontweight='bold', pad=10, loc='left')
    
    ax4.axvspan(-20, 0, alpha=0.1, color='blue', label='_nolegend_')
    ax4.axvspan(60, 80, alpha=0.1, color='red', label='_nolegend_')
    ax4.text(-10, 112, 'Cold Climate\nChallenge', fontsize=8, ha='center', color='blue', alpha=0.7)
    ax4.text(70, 112, 'Hot Climate\nChallenge', fontsize=8, ha='center', color='red', alpha=0.7)
    
    fig.suptitle('Figure 6. Electrolyte System Optimization and Environmental Factors', 
                 fontsize=13, fontweight='bold', y=0.98)
    
    plt.savefig('Figure6_Electrolyte_System.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print('✓ Figure 6 created')

if __name__ == '__main__':
    create_figure5()
    create_figure6()
    print('\n✓ All figures generated!')
