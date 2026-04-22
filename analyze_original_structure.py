"""
Analyze original CBES.docx structure:
- Map each paragraph to its section
- Identify "literature-listing" paragraphs (Author et al. / [ref] patterns)
- Flag paragraphs that could be condensed into synthetic summaries
- Generate a detailed report for the author to decide what to move to SI
"""
import re
import os
from docx import Document

SOURCE = r"C:\Users\wanzh\Desktop\研究进展\水泥储能投稿文件\CBES.docx"
REPORT = r"C:\Users\wanzh\Desktop\研究进展\水泥储能投稿文件\CBES_Streamline_Report.txt"

SECTION_MARKERS = [
    (re.compile(r'(?i)introduction'), '1. Introduction'),
    (re.compile(r'(?i)conduction\s+mechanism'), '2. Conduction Mechanisms'),
    (re.compile(r'(?i)structural\s+batter'), '3. Structural Batteries'),
    (re.compile(r'(?i)structural\s+supercapacitor'), '4. Structural Supercapacitors'),
    (re.compile(r'(?i)engineering\s+fabrication'), '5. Engineering Fabrication'),
    (re.compile(r'(?i)application\s+prospect'), '6. Application Prospects'),
    (re.compile(r'(?i)economic\s+and\s+environmental'), '7. Economic and Environmental Impact'),
    (re.compile(r'(?i)conclusion'), '8. Conclusion and Outlook'),
    (re.compile(r'(?i)reference'), 'References'),
    (re.compile(r'(?i)conflict|data\s+availability|acknowledge'), 'Back Matter'),
]

LITERATURE_PATTERNS = [
    re.compile(r'\b[A-Z][a-z]+\s+et\s+al\.'),
    re.compile(r'\[\d+\]'),
    re.compile(r'\b[A-Z][a-z]+\s+and\s+[A-Z][a-z]+'),
    re.compile(r'\(\d{4}\)'),
    re.compile(r'\breported\b|\bdemonstrated\b|\binvestigated\b|\bproposed\b|\bdeveloped\b|\bfabricated\b'),
]

SUBSECTION_PATTERNS = [
    re.compile(r'^\d+\.\d+'),
    re.compile(r'^\d+\.\d+\.\d+'),
]


def classify_paragraph(text):
    if not text.strip():
        return 'empty', 0

    is_heading = False
    for pat, name in SECTION_MARKERS:
        if pat.search(text.strip()):
            return 'section_heading', 0

    if SUBSECTION_PATTERNS[0].match(text.strip()):
        return 'subsection_heading', 0
    if SUBSECTION_PATTERNS[1].match(text.strip()):
        return 'subsubsection_heading', 0

    if len(text.strip()) < 30:
        return 'short', 0

    lit_score = 0
    for pat in LITERATURE_PATTERNS:
        if pat.search(text):
            lit_score += 1

    if lit_score >= 3:
        return 'literature_listing', lit_score
    elif lit_score >= 2:
        return 'literature_mixed', lit_score
    elif lit_score == 1:
        return 'literature_light', lit_score
    else:
        return 'synthetic', 0


def analyze():
    doc = Document(SOURCE)
    paragraphs = [p.text for p in doc.paragraphs]
    total = len(paragraphs)

    current_section = "Front Matter"
    section_data = {}
    para_details = []

    for i, text in enumerate(paragraphs):
        ptype, lit_score = classify_paragraph(text)

        for pat, name in SECTION_MARKERS:
            if pat.search(text.strip()):
                current_section = name
                ptype = 'section_heading'
                break

        if SUBSECTION_PATTERNS[0].match(text.strip()) and ptype != 'section_heading':
            ptype = 'subsection_heading'

        if current_section not in section_data:
            section_data[current_section] = {
                'total': 0,
                'literature_listing': 0,
                'literature_mixed': 0,
                'literature_light': 0,
                'synthetic': 0,
                'headings': 0,
                'short': 0,
                'empty': 0,
                'para_indices': [],
            }

        section_data[current_section]['total'] += 1
        section_data[current_section][ptype] = section_data[current_section].get(ptype, 0) + 1
        section_data[current_section]['para_indices'].append(i)

        para_details.append({
            'idx': i,
            'section': current_section,
            'type': ptype,
            'lit_score': lit_score,
            'text': text[:120] + '...' if len(text) > 120 else text,
        })

    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CBES ORIGINAL PAPER STRUCTURE ANALYSIS\n")
        f.write("Purpose: Identify paragraphs for streamlining (main text -> SI)\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. SECTION-BY-SECTION BREAKDOWN\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Section':<35} {'Total':>5} {'LitList':>7} {'LitMix':>6} {'Synth':>5} {'Head':>4}\n")
        f.write("-" * 60 + "\n")

        for section, data in section_data.items():
            f.write(f"{section:<35} {data['total']:>5} {data['literature_listing']:>7} "
                   f"{data['literature_mixed']:>6} {data['synthetic']:>5} {data['headings']:>4}\n")

        f.write("-" * 60 + "\n")
        total_lit = sum(d['literature_listing'] for d in section_data.values())
        total_mix = sum(d['literature_mixed'] for d in section_data.values())
        total_synth = sum(d['synthetic'] for d in section_data.values())
        f.write(f"{'TOTAL':<35} {total:>5} {total_lit:>7} {total_mix:>6} {total_synth:>5}\n\n")

        f.write("2. STREAMLINING PRIORITY (sections with most literature-listing paragraphs)\n")
        f.write("-" * 60 + "\n")
        priority_sections = sorted(
            [(s, d['literature_listing'], d['literature_mixed'], d['total'])
             for s, d in section_data.items()
             if d['literature_listing'] > 0 or d['literature_mixed'] > 0],
            key=lambda x: x[1] + x[2], reverse=True
        )
        for section, lit, mix, tot in priority_sections:
            condensable = lit + mix
            potential_saving = condensable * 0.6
            f.write(f"  {section:<35} LitList={lit} LitMix={mix} "
                   f"Condensable={condensable} "
                   f"Potential reduction: ~{potential_saving:.0f} paragraphs\n")

        f.write("\n3. DETAILED PARAGRAPH-BY-PARAGRAPH ANALYSIS\n")
        f.write("   (Only showing literature_listing and literature_mixed paragraphs)\n")
        f.write("-" * 80 + "\n\n")

        current = None
        for p in para_details:
            if p['type'] in ('literature_listing', 'literature_mixed'):
                if p['section'] != current:
                    current = p['section']
                    f.write(f"\n  === {current} ===\n\n")
                f.write(f"  [P{p['idx']:>3}] [{p['type']}] (lit_score={p['lit_score']})\n")
                f.write(f"         {p['text']}\n\n")

        f.write("\n4. RECOMMENDED ACTIONS\n")
        f.write("=" * 80 + "\n\n")

        f.write("ACTION 1: Condense Section 4 (Structural Supercapacitors)\n")
        f.write("  - This section has the most literature-listing paragraphs\n")
        f.write("  - Strategy: Group by electrode material type (CB, CNT, AC, Graphene)\n")
        f.write("  - For each type: 1 synthetic paragraph summarizing key findings\n")
        f.write("  - Move individual study details to SI Section S4\n")
        f.write("  - Expected reduction: ~15-20 paragraphs -> ~5-8 paragraphs\n\n")

        f.write("ACTION 2: Reorganize Section 6 (Application Prospects)\n")
        f.write("  - Currently organized by application type (4 subsections)\n")
        f.write("  - R3 criticism: lacks unified framework\n")
        f.write("  - Strategy: Reorganize by TRL level\n")
        f.write("    - Near-term (TRL 5-7): de-icing, SHM\n")
        f.write("    - Medium-term (TRL 3-5): EV charging, off-grid\n")
        f.write("    - Long-term (TRL 1-3): smart grid\n")
        f.write("  - Move detailed case studies to SI Section S6\n")
        f.write("  - Expected reduction: ~10-15 paragraphs -> ~5-7 paragraphs\n\n")

        f.write("ACTION 3: Consolidate Section 2 / Section 4.4 overlap\n")
        f.write("  - R3 identified redundancy between these sections\n")
        f.write("  - Section 2: intrinsic mechanisms (keep)\n")
        f.write("  - Section 4.4: engineering optimization (keep but cross-reference S2)\n")
        f.write("  - Add explicit statement: 'For detailed mechanism analysis, see Section 2'\n")
        f.write("  - Expected reduction: ~3-5 paragraphs\n\n")

        f.write("ACTION 4: Reduce reference-listing style in Section 3\n")
        f.write("  - Section 3 (Structural Batteries) also has significant lit-listing\n")
        f.write("  - Strategy: Group by battery chemistry (Ni-Fe, Zn, Cu, Mg)\n")
        f.write("  - For each chemistry: 1 comparative paragraph\n")
        f.write("  - Move individual study details to SI Section S3\n")
        f.write("  - Expected reduction: ~8-12 paragraphs -> ~4-6 paragraphs\n\n")

        f.write("ACTION 5: Trim Section 8 (Conclusion and Outlook)\n")
        f.write("  - 23 paragraphs is unusually long for a conclusion\n")
        f.write("  - Typical review conclusion: 5-10 paragraphs\n")
        f.write("  - Strategy: Keep key conclusions + outlook, move extended discussion to SI\n")
        f.write("  - Expected reduction: ~10-13 paragraphs -> ~5-8 paragraphs\n\n")

        total_body = sum(d['total'] for s, d in section_data.items()
                        if s not in ('References', 'Back Matter', 'Front Matter'))
        total_condensable = total_lit + total_mix
        estimated_saving = total_condensable * 0.6
        f.write(f"SUMMARY:\n")
        f.write(f"  Total body paragraphs (excl. refs/front/back): {total_body}\n")
        f.write(f"  Literature-listing paragraphs: {total_lit}\n")
        f.write(f"  Literature-mixed paragraphs: {total_mix}\n")
        f.write(f"  Total condensable: {total_condensable}\n")
        f.write(f"  Estimated paragraph reduction: ~{estimated_saving:.0f}\n")
        f.write(f"  Projected body paragraphs after streamlining: ~{total_body - estimated_saving:.0f}\n")
        f.write(f"  Reduction: ~{estimated_saving/total_body*100:.0f}%\n")

    print(f"Report saved to: {REPORT}")
    print(f"\nKey findings:")
    print(f"  Total paragraphs: {total}")
    print(f"  Literature-listing: {total_lit}")
    print(f"  Literature-mixed: {total_mix}")
    print(f"  Condensable: {total_condensable}")
    print(f"  Estimated reduction: ~{estimated_saving:.0f} paragraphs ({estimated_saving/total_body*100:.0f}%)")


if __name__ == '__main__':
    analyze()
