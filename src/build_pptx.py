"""build_pptx - Genera la presentacion ejecutiva del CCV.

Hereda template TFM/Presentacion_TFM.pptx (fondo dark + master SDC),
elimina las 6 slides de ejemplo, monta 27 slides nuevas con:
- Font Chakra Petch (coherente con outputs/viz)
- Texto blanco sobre fondo dark
- Visuales del repo dominando la composicion (70% visual / 30% texto)
- Logo JO bottom-right como firma personal
- Tono humano, minimalista, alineado con doc TFM
"""
from __future__ import annotations
from pathlib import Path
from copy import deepcopy

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

REPO = Path(__file__).resolve().parents[1]
TPL  = REPO / 'TFM' / 'Presentacion_TFM.pptx'
OUT  = REPO / 'TFM' / 'CCV_Presentacion_Ejecutiva.pptx'
VIZ  = REPO / 'outputs' / 'viz'
LOGO = REPO / 'outputs' / 'assets' / 'logo.png'

# Paleta (sobre fondo dark heredado del template)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
SOFT    = RGBColor(0xD1, 0xD5, 0xDB)
BLUE    = RGBColor(0x5D, 0xA8, 0xFF)
ORANGE  = RGBColor(0xFA, 0x89, 0x1E)
PURPLE  = RGBColor(0xB0, 0x84, 0xFF)
FONT    = 'Chakra Petch'


def _clear_slides(prs: Presentation) -> None:
    sldIdLst = prs.slides._sldIdLst
    for sldId in list(sldIdLst):
        rId = sldId.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
        prs.part.drop_rel(rId)
        sldIdLst.remove(sldId)


def _txt(slide, x, y, w, h, text, size, color=WHITE, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    tf.vertical_anchor = anchor
    # primera linea reutiliza el paragraph[0]
    lines = text.split('\n') if isinstance(text, str) else text
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_before = Pt(0)
        p.space_after = Pt(4)
        r = p.add_run()
        r.text = line
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
    return tb


def _img(slide, path, x, y, w=None, h=None):
    kw = {}
    if w is not None: kw['width']  = Inches(w)
    if h is not None: kw['height'] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def _jo_logo(slide):
    """logo JO bottom-right, pequeno, como firma."""
    return _img(slide, LOGO, x=11.85, y=6.95, w=1.30)


def _page_num(slide, n, total):
    _txt(slide, x=0.4, y=7.10, w=1.5, h=0.30,
         text=f'{n:02d} / {total:02d}', size=10, color=SOFT, align=PP_ALIGN.LEFT)


def _section_tag(slide, tag):
    _txt(slide, x=0.4, y=0.30, w=4, h=0.30,
         text=tag.upper(), size=10, color=BLUE, bold=True, align=PP_ALIGN.LEFT)


def _blank_slide(prs):
    # layout 0 = OBJECT (hereda fondo dark del template)
    s = prs.slides.add_slide(prs.slide_layouts[0])
    # quitar placeholders heredados
    for sh in list(s.placeholders):
        sp = sh._element
        sp.getparent().remove(sp)
    return s


def build():
    prs = Presentation(str(TPL))
    _clear_slides(prs)

    total = 27

    # ============ 1. PORTADA ============
    s = _blank_slide(prs)
    _txt(s, 0.7, 2.3, 11.9, 0.9, 'CAUSAL CLUTCH VALUE',
         size=52, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    _txt(s, 0.7, 3.3, 11.9, 0.6, 'Cuantificación causal del clutch en el fútbol de élite',
         size=22, color=BLUE, align=PP_ALIGN.CENTER)
    _txt(s, 0.7, 4.4, 11.9, 0.4, 'Trabajo Fin de Máster',
         size=14, color=SOFT, align=PP_ALIGN.CENTER)
    _txt(s, 0.7, 4.9, 11.9, 0.4, 'Máster en Big Data Aplicado al Scouting Deportivo',
         size=14, color=SOFT, align=PP_ALIGN.CENTER)
    _txt(s, 0.7, 5.4, 11.9, 0.4, 'Sports Data Campus · 2025-2026',
         size=14, color=SOFT, align=PP_ALIGN.CENTER)
    _txt(s, 0.7, 6.15, 11.9, 0.4, 'Jaime Oriol Goicoechea',
         size=18, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    _img(s, LOGO, x=5.97, y=6.55, w=1.40)

    # ============ 2. HOOK ============
    s = _blank_slide(prs)
    _section_tag(s, 'Motivación')
    _txt(s, 0.7, 1.4, 11.9, 1.1, '"Mbappé en la Final"',
         size=44, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    _txt(s, 0.7, 2.8, 11.9, 0.6, 'Tres goles en una final. Dos en menos de dos minutos.',
         size=20, color=SOFT, align=PP_ALIGN.CENTER)
    _txt(s, 0.7, 3.5, 11.9, 0.6, 'La intuición dice que respondió al shock. La pregunta es si se puede medir.',
         size=20, color=SOFT, align=PP_ALIGN.CENTER)
    _txt(s, 0.7, 5.0, 11.9, 0.7,
         'El clutch existe en la cabeza del aficionado.',
         size=22, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
    _txt(s, 0.7, 5.6, 11.9, 0.7,
         '¿Existe en los datos?',
         size=22, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
    _jo_logo(s); _page_num(s, 2, total)

    # ============ 3. GAP ============
    s = _blank_slide(prs)
    _section_tag(s, 'Problema')
    _txt(s, 0.7, 0.9, 11.9, 0.8, 'El gap del scouting actual',
         size=34, color=WHITE, bold=True)
    _txt(s, 0.7, 2.2, 11.9, 0.6,
         'Los rankings públicos del "jugador que aparece cuando importa" parten de eventing puro.',
         size=18, color=SOFT)
    _txt(s, 0.7, 3.0, 11.9, 0.6,
         'Sin tracking sincronizado. Sin separación entre lo individual y el empuje colectivo.',
         size=18, color=SOFT)
    _txt(s, 0.7, 4.3, 11.9, 0.6,
         'Resultado: un jugador con buena estadística post-gol suele ser el que',
         size=18, color=SOFT)
    _txt(s, 0.7, 4.9, 11.9, 0.6,
         'estaba en un equipo que ya estaba dominando el partido.',
         size=18, color=SOFT)
    _txt(s, 0.7, 6.0, 11.9, 0.7,
         'El clutch verdadero requiere descontar lo que hace su bloque.',
         size=20, color=ORANGE, bold=True)
    _jo_logo(s); _page_num(s, 3, total)

    # ============ 4. OBJETIVO ============
    s = _blank_slide(prs)
    _section_tag(s, 'Objetivos')
    _txt(s, 0.7, 0.9, 11.9, 0.8, 'Objetivo general', size=34, color=WHITE, bold=True)
    _txt(s, 0.7, 2.5, 11.9, 2.0,
         'Estimar el efecto causal del shock emocional del gol sobre el rendimiento individual,',
         size=22, color=WHITE)
    _txt(s, 0.7, 3.2, 11.9, 0.8,
         'residualizado contra el delta del propio bloque,',
         size=22, color=WHITE)
    _txt(s, 0.7, 3.9, 11.9, 0.8,
         'en cuatro canales y tres contextos.',
         size=22, color=WHITE)
    _txt(s, 0.7, 5.4, 11.9, 0.7,
         'Output canónico: tabla scout-facing con un jugador por fila +',
         size=18, color=BLUE)
    _txt(s, 0.7, 6.0, 11.9, 0.7,
         'intervalos de credibilidad bayesianos por canal y contexto.',
         size=18, color=BLUE)
    _jo_logo(s); _page_num(s, 4, total)

    # ============ 5. 6 OEs ============
    s = _blank_slide(prs)
    _section_tag(s, 'Objetivos')
    _txt(s, 0.7, 0.9, 11.9, 0.8, 'Seis objetivos específicos',
         size=34, color=WHITE, bold=True)
    oes = [
        ('OE1', 'Backbone temporal: WP bayesiana + leverage + proximidad eliminación'),
        ('OE2', 'PSxG calibrado + corpus de near-miss como control cuasi-experimental'),
        ('OE3', 'Cuatro canales: ofensivo, defensivo, off-ball, físico'),
        ('OE4', 'Identificación causal dual: DiD within-player + AIPW cross-fitting'),
        ('OE5', 'CATE jerárquico bayesiano multivariate (NUTS HMC + LKJ)'),
        ('OE6', 'Ensamblaje scout-facing con cuantificación de incertidumbre por jugador'),
    ]
    for i, (k, v) in enumerate(oes):
        y = 2.2 + i * 0.75
        _txt(s, 0.7, y, 1.4, 0.6, k, size=20, color=BLUE, bold=True)
        _txt(s, 2.1, y, 10.5, 0.6, v, size=18, color=WHITE)
    _jo_logo(s); _page_num(s, 5, total)

    # ============ 6. TRES FUENTES ============
    s = _blank_slide(prs)
    _section_tag(s, 'Datos')
    _txt(s, 0.7, 0.9, 11.9, 0.8, 'Tres fuentes, una pregunta',
         size=34, color=WHITE, bold=True)
    cols = [
        ('PFF FC',     'World Cup Qatar 2022',  'Target (sagrado)',  '64 partidos · tracking 25Hz · grades por evento'),
        ('Wyscout',    '2017/18 open dataset',   'Backbone temporal', '1941 partidos · 5 grandes ligas + WC18 + Euro16'),
        ('StatsBomb',  'Open data filtrado',     'PSxG + cross val',  '136 partidos · Euro20 + Euro24 + BL 23/24'),
    ]
    for i, (k, sub, role, vol) in enumerate(cols):
        x = 0.7 + i * 4.2
        _txt(s, x, 2.3, 4.0, 0.7, k, size=26, color=BLUE, bold=True)
        _txt(s, x, 3.0, 4.0, 0.5, sub, size=14, color=SOFT)
        _txt(s, x, 3.8, 4.0, 0.5, role.upper(), size=12, color=ORANGE, bold=True)
        _txt(s, x, 4.4, 4.0, 1.8, vol, size=14, color=WHITE)
    _txt(s, 0.7, 6.3, 11.9, 0.5,
         'Las tres extraídas a parquet con tipos anidados y verificación round-trip lossless.',
         size=14, color=SOFT, align=PP_ALIGN.CENTER)
    _jo_logo(s); _page_num(s, 6, total)

    # ============ 7. POR QUE PFF ============
    s = _blank_slide(prs)
    _section_tag(s, 'Datos')
    _txt(s, 0.7, 0.9, 11.9, 0.8, 'Por qué PFF FC',
         size=34, color=WHITE, bold=True)
    _txt(s, 0.7, 1.8, 11.9, 0.5,
         'Único corpus público que combina los tres ingredientes que la pregunta exige:',
         size=16, color=SOFT)
    ings = [
        '1', 'Tracking continuo full-pitch de los 22 jugadores a 25 Hz',
        '2', 'Sincronización exacta con el eventing',
        '3', 'Cobertura completa de fase eliminatoria con prórroga',
    ]
    for i in range(3):
        y = 2.7 + i * 0.85
        _txt(s, 1.0, y, 0.6, 0.7, ings[2*i],     size=32, color=ORANGE, bold=True)
        _txt(s, 1.8, y+0.10, 10.8, 0.6, ings[2*i+1], size=20, color=WHITE)
    _txt(s, 0.7, 5.8, 11.9, 0.5,
         'PFF añade además grades humanos por evento — prior informativo opcional del CATE.',
         size=14, color=BLUE)
    _txt(s, 0.7, 6.3, 11.9, 0.5,
         'StatsBomb 360 cubre freeze-frames en eventos puntuales, no aporta tracking continuo entre eventos.',
         size=14, color=SOFT)
    _jo_logo(s); _page_num(s, 7, total)

    # ============ 8. VOLUMEN ============
    s = _blank_slide(prs)
    _section_tag(s, 'Datos')
    _txt(s, 0.7, 0.9, 11.9, 0.8, 'El corpus de trabajo en cifras',
         size=34, color=WHITE, bold=True)
    cells = [
        ('64',   'partidos del Mundial'),
        ('172',  'shocks de gol (120 grupos + 52 KO)'),
        ('70',   'near-misses pre-registrados (5 tipologías)'),
        ('511',  'jugadores con minutos en campo'),
        ('233',  'jugadores con ≥ 270 min (muestra fiable)'),
        ('166',  'jugadores con grades PFF validación externa'),
    ]
    for i, (n, label) in enumerate(cells):
        row, col = divmod(i, 3)
        x = 0.7 + col * 4.2
        y = 2.3 + row * 2.2
        _txt(s, x, y, 4.0, 1.2, n, size=60, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
        _txt(s, x, y+1.3, 4.0, 0.7, label, size=14, color=WHITE, align=PP_ALIGN.CENTER)
    _jo_logo(s); _page_num(s, 8, total)

    # ============ 9. PIPELINE DAG ============
    s = _blank_slide(prs)
    _section_tag(s, 'Pipeline')
    _txt(s, 0.7, 0.9, 11.9, 0.8, 'Pipeline en 6 fases',
         size=34, color=WHITE, bold=True)
    _img(s, VIZ / 'fig_cap4_pipeline_dag.jpg', x=0.6, y=1.95, w=12.1)
    _jo_logo(s); _page_num(s, 9, total)

    # ============ 10. MAPA CONCEPTUAL ============
    s = _blank_slide(prs)
    _section_tag(s, 'Marco')
    _txt(s, 0.7, 0.9, 11.9, 0.8, 'Mapa conceptual del CCV',
         size=34, color=WHITE, bold=True)
    _txt(s, 0.7, 1.65, 11.9, 0.5,
         'Tipo de shock × ¿el jugador rompe del bloque? × ¿reacción positiva? → 4 canales',
         size=14, color=SOFT)
    _img(s, VIZ / 'fig_cap4_mapa_conceptual.png', x=1.0, y=2.30, w=11.3)
    _jo_logo(s); _page_num(s, 10, total)

    # ============ 11. 5 CAPAS CAUSALES ============
    s = _blank_slide(prs)
    _section_tag(s, 'Marco')
    _txt(s, 0.7, 0.9, 11.9, 0.8, 'Cinco capas de identificación causal',
         size=34, color=WHITE, bold=True)
    _txt(s, 0.7, 1.65, 11.9, 0.5,
         'Cada capa retira un confusor distinto. El estimador del CCV emerge sólo cuando las cinco operan en secuencia.',
         size=14, color=SOFT)
    _img(s, VIZ / 'fig_cap4_capas_causales.png', x=2.4, y=2.20, w=8.5)
    _jo_logo(s); _page_num(s, 11, total)

    # ============ 12. DiD ============
    s = _blank_slide(prs)
    _section_tag(s, 'Identificación causal')
    _txt(s, 0.7, 0.9, 11.9, 0.8, 'Diferencias-en-diferencias within-player',
         size=30, color=WHITE, bold=True)
    _txt(s, 0.7, 1.85, 6.0, 0.5, 'Modelo TWFE:', size=16, color=BLUE, bold=True)
    _txt(s, 0.7, 2.4, 6.5, 1.0,
         'Y = αᵢₛ + γₜ−ₜₛ + τ · 1[t > tₛ] + ε',
         size=16, color=WHITE)
    _txt(s, 0.7, 3.3, 6.5, 0.5, 'Verificaciones independientes:', size=16, color=BLUE, bold=True)
    bullets = [
        '• Sun-Abraham 2021 (no contaminación entre cohortes)',
        '• Borusyak-Jaravel-Spiess 2024 (imputación contrafactual)',
        '• HonestDiD Rambachan-Roth 2023 (cotas de relajación)',
        '• Test F de pre-tendencias (Roth 2022)',
    ]
    for i, b in enumerate(bullets):
        _txt(s, 0.9, 3.85 + i*0.5, 6.3, 0.5, b, size=14, color=WHITE)
    _img(s, VIZ / 'event_study.png', x=7.4, y=1.8, w=5.5)
    _jo_logo(s); _page_num(s, 12, total)

    # ============ 13. AIPW ============
    s = _blank_slide(prs)
    _section_tag(s, 'Identificación causal')
    _txt(s, 0.7, 0.9, 11.9, 0.8, 'AIPW con cross-fitting',
         size=34, color=WHITE, bold=True)
    _txt(s, 0.7, 1.85, 11.9, 0.5,
         'Doblemente robusto asintótico bajo correcta especificación de un nuisance,',
         size=18, color=SOFT)
    _txt(s, 0.7, 2.4, 11.9, 0.5,
         'overlap positivo y consistencia n⁻¹ᐟ⁴ (Chernozhukov et al. 2018).',
         size=18, color=SOFT)
    _txt(s, 0.7, 3.5, 11.9, 0.6, 'Nuisance models', size=16, color=BLUE, bold=True)
    _txt(s, 0.9, 4.0, 11.5, 0.5, '• LightGBM 5-fold estratificado por partido', size=16, color=WHITE)
    _txt(s, 0.9, 4.5, 11.5, 0.5, '• Propensity score + outcome regression', size=16, color=WHITE)
    _txt(s, 0.7, 5.3, 11.9, 0.6, 'Comprobaciones complementarias', size=16, color=BLUE, bold=True)
    _txt(s, 0.9, 5.8, 11.5, 0.5, '• DML Partially Linear Regression', size=16, color=WHITE)
    _txt(s, 0.9, 6.3, 11.5, 0.5, '• DR-learner (Kennedy 2023) + RDD local-lineal sobre xG', size=16, color=WHITE)
    _jo_logo(s); _page_num(s, 13, total)

    # ============ 14. CATE BAYESIANO ============
    s = _blank_slide(prs)
    _section_tag(s, 'CATE')
    _txt(s, 0.7, 0.9, 11.9, 0.8, 'CATE jerárquico bayesiano multivariate',
         size=30, color=WHITE, bold=True)
    _txt(s, 0.7, 1.9, 6.0, 0.5, 'Estructura:', size=16, color=BLUE, bold=True)
    feats = [
        '• 4 canales conjuntamente (no separados)',
        '• Correlaciones cross-canal vía LKJ Cholesky',
        '• Priors informados por grades PFF (opcionales)',
        '• Parametrización non-centered (Betancourt 2015)',
    ]
    for i, f in enumerate(feats):
        _txt(s, 0.9, 2.4 + i*0.5, 6.3, 0.5, f, size=14, color=WHITE)
    _txt(s, 0.7, 4.8, 6.0, 0.5, 'Sampling:', size=16, color=BLUE, bold=True)
    samp = [
        '• NUTS HMC · 4 chains × 1000+1000',
        '• target_accept_prob = 0.95',
        '• 0 divergencias · 108/144 R-hat < 1.05',
        '• PPC 8/8 channels calibrados (KS p > 0.05)',
    ]
    for i, f in enumerate(samp):
        _txt(s, 0.9, 5.3 + i*0.45, 6.3, 0.5, f, size=14, color=WHITE)
    _img(s, VIZ / 'fig_cap6_cate_heterogeneity.png', x=7.4, y=1.9, w=5.5)
    _jo_logo(s); _page_num(s, 14, total)

    # ============ 15-18. 4 CANALES ============
    canales = [
        (15, 'Canal ofensivo',
            ['atomic-VAEP (CatBoost) sobre representación SPADL',
             'un-xPass (Robberechts 2023) para creatividad en pase',
             'AUC scores 0.83 / concedes 0.87 · un-xPass 0.83',
             'Entrenamiento: 456k acciones + 137k pases'],
            'scatter_ataque_marcar_encajar.png'),
        (16, 'Canal defensivo',
            ['VDEP estricto (Toda 2022): recovery + attacked',
             'exPress (Lee 2025): P(recuperación < 5s | presión)',
             'Atribución frame-nearest al defensor más cercano',
             'AUC VDEP 0.80/0.83 · exPress 0.62'],
            'fig_cap6_psxg_calibration.png'),
        (17, 'Canal off-ball',
            ['OBSO + C-OBSO (Spearman 2018 + Teranishi 2022)',
             'Pitch Control vectorizado sobre tracking 25 Hz',
             '64 partidos full-time 22 jugadores',
             'C-OBSO vs grade PFF: r = +0.29 (n=673, p<10⁻¹⁴)'],
            'ppcf_mbappe_2_2_final.png'),
        (18, 'Canal físico',
            ['Protocolo Bradley 2024 sobre velocidades suavizadas',
             'Filtro Butterworth fase cero · cap 11 m/s',
             'Residualización bayesiana jerárquica (SVI)',
             'Línea base personal modulada por curva temporal'],
            None),  # sin visual disponible
    ]
    for n, title, bullets, viz_file in canales:
        s = _blank_slide(prs)
        _section_tag(s, 'Canales')
        _txt(s, 0.7, 0.9, 11.9, 0.8, title, size=32, color=WHITE, bold=True)
        if viz_file:
            for i, b in enumerate(bullets):
                _txt(s, 0.7, 2.2 + i*0.7, 6.5, 0.6, '• ' + b, size=16, color=WHITE)
            _img(s, VIZ / viz_file, x=7.4, y=2.0, w=5.5)
        else:
            for i, b in enumerate(bullets):
                _txt(s, 0.7, 2.5 + i*0.9, 12.0, 0.7, '• ' + b, size=20, color=WHITE)
        _jo_logo(s); _page_num(s, n, total)

    # ============ 19. PSxG ANCLA ============
    s = _blank_slide(prs)
    _section_tag(s, 'Validación')
    _txt(s, 0.7, 0.9, 11.9, 0.8, 'PSxG: ancla técnica del near-miss',
         size=32, color=WHITE, bold=True)
    _txt(s, 0.7, 1.85, 6.0, 0.5,
         'LightGBM + Optuna 60 iter + isotonic',
         size=16, color=BLUE, bold=True)
    metrics = [
        ('AUC OOF',          '0.968'),
        ('AUC holdout WC22', '0.976'),
        ('vs StatsBomb xG',  '0.844'),
        ('ECE holdout',      '0.011'),
        ('Brier holdout',    '0.037'),
    ]
    for i, (k, v) in enumerate(metrics):
        y = 2.5 + i * 0.55
        _txt(s, 0.9, y, 4.0, 0.5, k, size=16, color=SOFT)
        _txt(s, 4.9, y, 2.0, 0.5, v, size=18, color=WHITE, bold=True, align=PP_ALIGN.RIGHT)
    _txt(s, 0.7, 5.6, 6.7, 0.6,
         'AUC alta justificable: PSxG es post-disparo (incluye end_y, end_z,',
         size=12, color=SOFT)
    _txt(s, 0.7, 6.0, 6.7, 0.6,
         'configuración 360 freeze-frame). xG StatsBomb es pre-disparo.',
         size=12, color=SOFT)
    _img(s, VIZ / 'fig_cap6_psxg_calibration.png', x=7.4, y=1.9, w=5.5)
    _jo_logo(s); _page_num(s, 19, total)

    # ============ 20. FICHA MBAPPE ============
    s = _blank_slide(prs)
    _section_tag(s, 'Output scout-facing')
    _txt(s, 0.7, 0.9, 11.9, 0.8, 'Ficha CCV — Kylian Mbappé',
         size=32, color=WHITE, bold=True)
    _txt(s, 0.7, 1.65, 11.9, 0.5,
         '#1 pressure-clutch del torneo · CATE +0.110 · P(β > 0) = 0.97',
         size=16, color=ORANGE, bold=True)
    _img(s, VIZ / 'radar_report_3870.png', x=1.5, y=2.20, w=10.3)
    _jo_logo(s); _page_num(s, 20, total)

    # ============ 21. SCATTER REMONTADOR x CERROJO ============
    s = _blank_slide(prs)
    _section_tag(s, 'Output scout-facing')
    _txt(s, 0.7, 0.9, 11.9, 0.8, 'Remontador × Cerrojo · 511 jugadores',
         size=30, color=WHITE, bold=True)
    _txt(s, 0.7, 1.65, 11.9, 0.5,
         'Dos índices ortogonales: respuesta al gol en contra vs. respuesta al gol a favor.',
         size=14, color=SOFT)
    _img(s, VIZ / 'scatter_remontador_cerrojo.png', x=1.5, y=2.10, w=10.3)
    _jo_logo(s); _page_num(s, 21, total)

    # ============ 22. ATE = 0 ============
    s = _blank_slide(prs)
    _section_tag(s, 'Resultados')
    _txt(s, 0.7, 0.9, 11.9, 0.8, 'ATE poblacional indistinguible de cero',
         size=30, color=WHITE, bold=True)
    _txt(s, 0.7, 1.95, 6.5, 0.6,
         'Ocho celdas canal × contexto.',
         size=18, color=WHITE)
    _txt(s, 0.7, 2.5, 6.5, 0.6,
         'Ninguna cruza la significatividad al 5%.',
         size=18, color=WHITE)
    _txt(s, 0.7, 3.4, 6.5, 0.6, 'Placebo 1000 perm + BH FDR:', size=14, color=BLUE, bold=True)
    _txt(s, 0.7, 3.9, 6.5, 0.6, 'ningún canal sale del null (p_FDR = 0.98)', size=14, color=WHITE)
    _txt(s, 0.7, 4.8, 6.5, 0.6, 'HonestDiD:', size=14, color=BLUE, bold=True)
    _txt(s, 0.7, 5.3, 6.5, 0.6, 'cotas contienen el cero en los 3 niveles', size=14, color=WHITE)
    _txt(s, 0.7, 6.0, 6.5, 0.7,
         'El efecto medio del shock no se distingue del ruido.',
         size=16, color=ORANGE, bold=True)
    _img(s, VIZ / 'fig_cap6_window_sensitivity.png', x=7.4, y=1.95, w=5.5)
    _jo_logo(s); _page_num(s, 22, total)

    # ============ 23. 4 PRESSURE-CLUTCH LEADERS ============
    s = _blank_slide(prs)
    _section_tag(s, 'Resultados')
    _txt(s, 0.7, 0.9, 11.9, 0.8, 'Cuatro pressure-clutch leaders identificados',
         size=30, color=WHITE, bold=True)
    _txt(s, 0.7, 1.7, 11.9, 0.5,
         'CATE pressure-clutch · P(β) ≥ 0.85 · 511 jugadores evaluados',
         size=14, color=SOFT)
    headers = ['Jugador', 'Selección', 'CATE', 'P(β > 0)']
    rows = [
        ('Kylian Mbappé',  'Francia',   '+0.110', '0.97'),
        ('Marcus Rashford', 'Inglaterra', '+0.050', '0.89'),
        ('Bukayo Saka',     'Inglaterra', '+0.050', '0.86'),
        ('Mohammed Muntari','Catar',     '−0.055', '0.12'),
    ]
    x0 = [0.9, 4.5, 7.5, 10.0]
    w0 = [3.5, 3.0, 2.5, 2.5]
    y = 2.7
    for i, h in enumerate(headers):
        _txt(s, x0[i], y, w0[i], 0.5, h, size=14, color=BLUE, bold=True)
    for j, row in enumerate(rows):
        yy = 3.3 + j * 0.7
        for i, v in enumerate(row):
            col = ORANGE if (i >= 2 and (v.startswith('+') or v.startswith('−'))) else WHITE
            _txt(s, x0[i], yy, w0[i], 0.6, v, size=18, color=col, bold=(i==0))
    _txt(s, 0.7, 6.5, 11.9, 0.5,
         'Muntari: el único negativo significativo. Cae bajo presión de eliminación inminente.',
         size=12, color=SOFT, align=PP_ALIGN.CENTER)
    _jo_logo(s); _page_num(s, 23, total)

    # ============ 24. 22 ofensivo ============
    s = _blank_slide(prs)
    _section_tag(s, 'Resultados')
    _txt(s, 0.7, 0.9, 11.9, 0.8, 'Ofensivo-tras-marcar: 22 jugadores significativos',
         size=28, color=WHITE, bold=True)
    _txt(s, 0.7, 1.85, 11.9, 0.5,
         'La celda canal × contexto con más señal individual del torneo.',
         size=14, color=SOFT)
    _txt(s, 0.7, 3.0, 5.5, 1.0, '5', size=80, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
    _txt(s, 0.7, 4.3, 5.5, 0.6, 'al alza', size=20, color=WHITE, align=PP_ALIGN.CENTER)
    _txt(s, 0.7, 4.9, 5.5, 0.5, '(asciende su producción ofensiva)', size=12, color=SOFT, align=PP_ALIGN.CENTER)
    _txt(s, 7.1, 3.0, 5.5, 1.0, '17', size=80, color=ORANGE, bold=True, align=PP_ALIGN.CENTER)
    _txt(s, 7.1, 4.3, 5.5, 0.6, 'a la baja', size=20, color=WHITE, align=PP_ALIGN.CENTER)
    _txt(s, 7.1, 4.9, 5.5, 0.5, '(se relaja tras marcar)', size=12, color=SOFT, align=PP_ALIGN.CENTER)
    _txt(s, 0.7, 6.2, 11.9, 0.7,
         'Signos opuestos en magnitudes similares: el promedio cancela y el ATE colapsa a cero.',
         size=16, color=BLUE, align=PP_ALIGN.CENTER)
    _jo_logo(s); _page_num(s, 24, total)

    # ============ 25. CONCLUSIÓN CLAVE ============
    s = _blank_slide(prs)
    _section_tag(s, 'Conclusión')
    _txt(s, 0.7, 1.5, 11.9, 1.0,
         'ATE nulo no implica efecto nulo.',
         size=40, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    _txt(s, 0.7, 2.8, 11.9, 1.0,
         'El clutch vive en la heterogeneidad individual,',
         size=28, color=BLUE, align=PP_ALIGN.CENTER)
    _txt(s, 0.7, 3.5, 11.9, 1.0,
         'no en la media poblacional.',
         size=28, color=BLUE, align=PP_ALIGN.CENTER)
    _txt(s, 0.7, 5.3, 11.9, 0.6,
         'El framework permite identificar QUIÉN responde al shock,',
         size=18, color=SOFT, align=PP_ALIGN.CENTER)
    _txt(s, 0.7, 5.9, 11.9, 0.6,
         'EN QUÉ CANAL y CON CUÁNTA CERTEZA.',
         size=18, color=SOFT, align=PP_ALIGN.CENTER)
    _jo_logo(s); _page_num(s, 25, total)

    # ============ 26. LIMITACIONES + FUTURO ============
    s = _blank_slide(prs)
    _section_tag(s, 'Limitaciones · Líneas futuras')
    _txt(s, 0.7, 0.9, 11.9, 0.8, 'Hasta dónde llega y por dónde sigue',
         size=30, color=WHITE, bold=True)
    _txt(s, 0.7, 1.95, 6.0, 0.6, 'Limitaciones', size=20, color=ORANGE, bold=True)
    lims = [
        '• Un único torneo (Mundial Qatar 2022)',
        '• 172 shocks → CATE individual fiable, no a partida',
        '• 70 near-misses → triangulación agregada, no por jugador',
        '• Building blocks no oficiales: re-implementación propia validada vs paper',
    ]
    for i, b in enumerate(lims):
        _txt(s, 0.9, 2.5 + i*0.6, 6.0, 0.6, b, size=14, color=WHITE)
    _txt(s, 7.1, 1.95, 6.0, 0.6, 'Líneas futuras', size=20, color=BLUE, bold=True)
    fut = [
        '• Más torneos KO (Eurocopa, Champions, World Cup 2026)',
        '• Corpus near-miss ampliado para verificación por jugador',
        '• Más catálogos de shock (sustituciones, sup./inf. numérica)',
        '• Portabilidad a "clutch del bloque" a nivel equipo',
    ]
    for i, b in enumerate(fut):
        _txt(s, 7.3, 2.5 + i*0.6, 6.0, 0.6, b, size=14, color=WHITE)
    _jo_logo(s); _page_num(s, 26, total)

    # ============ 27. GRACIAS ============
    s = _blank_slide(prs)
    _txt(s, 0.7, 2.5, 11.9, 1.2, 'Gracias',
         size=72, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    _txt(s, 0.7, 4.2, 11.9, 0.6,
         'Repositorio público con el pipeline E2E completo + documento TFM:',
         size=16, color=SOFT, align=PP_ALIGN.CENTER)
    _txt(s, 0.7, 4.85, 11.9, 0.6, 'github.com/jaime-oriol/CCV',
         size=22, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
    _img(s, LOGO, x=5.97, y=5.85, w=1.40)
    _txt(s, 0.7, 6.85, 11.9, 0.4, 'Jaime Oriol Goicoechea · 2026',
         size=11, color=SOFT, align=PP_ALIGN.CENTER)

    prs.save(str(OUT))
    print(f'guardado: {OUT.relative_to(REPO)}')
    print(f'  slides: {len(prs.slides)}')


if __name__ == '__main__':
    build()
