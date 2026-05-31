"""build_pptx - Presentacion ejecutiva del CCV.

Diseno LIGHT OPTA matching outputs/viz + PDF del TFM:
- Fondo blanco limpio
- Texto negro/azul sobre blanco (alto contraste, legible)
- Font Chakra Petch (coherente con todas las viz)
- Charts dominando la composicion (cada uno se entiende solo)
- Logo JO firma grande
- Small SDC logo como brand attribution

Modo de uso pensado: el profesor LEE el deck como version compacta
del documento TFM. No requiere narracion oral.
"""
from __future__ import annotations
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from lxml import etree

REPO = Path(__file__).resolve().parents[1]
OUT  = REPO / 'TFM' / 'CCV_Presentacion_Ejecutiva.pptx'
VIZ  = REPO / 'outputs' / 'viz'
LOGO = REPO / 'outputs' / 'assets' / 'logo.png'
# SDC branding via texto en portada/cierre (sin logo embebido, evita dependencia de path)

# Paleta LIGHT OPTA (matches viz)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
NAVY    = RGBColor(0x1E, 0x3A, 0x8A)  # azul profundo titles
INK     = RGBColor(0x1A, 0x1A, 0x1A)  # negro suave body
GRAY    = RGBColor(0x55, 0x5B, 0x6E)  # gris caption
SOFT    = RGBColor(0x9C, 0xA3, 0xAF)  # gris very soft
ORANGE  = RGBColor(0xEA, 0x58, 0x0C)  # acento clave
BLUE_LT = RGBColor(0x3B, 0x82, 0xF6)  # azul accents
SEP     = RGBColor(0xE5, 0xE7, 0xEB)  # gris linea separadora
FONT    = 'Chakra Petch'


def _white_bg(slide):
    """Forzar fondo blanco solido (override default master)."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = WHITE


def _txt(slide, x, y, w, h, text, size, color=INK, bold=False, italic=False,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line_spacing=1.15):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    tf.vertical_anchor = anchor
    lines = text.split('\n') if isinstance(text, str) else text
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        p.space_before = Pt(0)
        p.space_after = Pt(2)
        r = p.add_run()
        r.text = line
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
    return tb


def _img(slide, path, x, y, w=None, h=None):
    kw = {}
    if w is not None: kw['width']  = Inches(w)
    if h is not None: kw['height'] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def _line(slide, x, y, w, color=NAVY, weight_pt=2):
    ln = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                 Inches(x), Inches(y), Inches(w), Pt(weight_pt))
    ln.fill.solid(); ln.fill.fore_color.rgb = color
    ln.line.fill.background()
    return ln


def _footer(slide, n, total):
    """Footer estandar: logo JO firma grande izquierda + numero pagina derecha."""
    # logo JO grande izquierda
    _img(slide, LOGO, x=0.45, y=6.85, w=1.85)
    # numero pagina derecha
    _txt(slide, x=12.10, y=7.05, w=0.90, h=0.30,
         text=f'{n:02d} / {total:02d}', size=10, color=SOFT,
         align=PP_ALIGN.RIGHT)


def _section_chip(slide, tag):
    """Tag de seccion top-right."""
    _txt(slide, x=10.6, y=0.40, w=2.5, h=0.30,
         text=tag.upper(), size=10, color=NAVY, bold=True,
         align=PP_ALIGN.RIGHT)


def _title(slide, text, size=34, color=NAVY):
    _txt(slide, x=0.6, y=0.55, w=11.7, h=0.90,
         text=text, size=size, color=color, bold=True)
    _line(slide, x=0.6, y=1.35, w=0.7, color=ORANGE, weight_pt=3)


def _blank(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])  # Layout 6 = Blank
    _white_bg(s)
    return s


def build():
    # presentacion limpia desde cero (sin heredar dark template)
    prs = Presentation()
    prs.slide_width  = Inches(13.333)
    prs.slide_height = Inches(7.500)

    total = 27

    # ============ 1. PORTADA ============
    s = _blank(prs)
    _line(s, x=4.5, y=1.65, w=4.3, color=ORANGE, weight_pt=4)
    _txt(s, 0.5, 1.95, 12.3, 1.0, 'CAUSAL CLUTCH VALUE',
         size=58, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    _txt(s, 0.5, 3.05, 12.3, 0.6,
         'Cuantificación causal del clutch en el fútbol de élite',
         size=22, color=BLUE_LT, align=PP_ALIGN.CENTER)
    _txt(s, 0.5, 4.20, 12.3, 0.5,
         'Trabajo Fin de Máster',
         size=15, color=GRAY, align=PP_ALIGN.CENTER)
    _txt(s, 0.5, 4.70, 12.3, 0.4,
         'Máster en Big Data Aplicado al Scouting Deportivo',
         size=13, color=GRAY, align=PP_ALIGN.CENTER)
    _txt(s, 0.5, 5.10, 12.3, 0.4,
         'Sports Data Campus · 2025-2026',
         size=13, color=GRAY, align=PP_ALIGN.CENTER)
    _img(s, LOGO, x=5.42, y=5.85, w=2.50)

    # ============ 2. RESUMEN EJECUTIVO ============
    s = _blank(prs)
    _section_chip(s, 'Resumen')
    _title(s, 'En una página')
    _txt(s, 0.6, 1.8, 12.1, 0.55,
         'Pregunta', size=16, color=ORANGE, bold=True)
    _txt(s, 0.6, 2.30, 12.1, 0.55,
         '¿Puede medirse causalmente la respuesta individual del jugador al shock emocional del gol,',
         size=17, color=INK)
    _txt(s, 0.6, 2.75, 12.1, 0.55,
         'separada del empuje colectivo de su propio equipo?',
         size=17, color=INK)
    _txt(s, 0.6, 3.55, 12.1, 0.55,
         'Marco', size=16, color=ORANGE, bold=True)
    _txt(s, 0.6, 4.05, 12.1, 0.55,
         'CCV: pipeline de 6 fases con identificación causal en 5 capas',
         size=17, color=INK)
    _txt(s, 0.6, 4.50, 12.1, 0.55,
         '(DiD within-player + AIPW cross-fitting + CATE jerárquico bayesiano multivariate).',
         size=17, color=INK)
    _txt(s, 0.6, 5.30, 12.1, 0.55,
         'Hallazgo', size=16, color=ORANGE, bold=True)
    _txt(s, 0.6, 5.80, 12.1, 0.55,
         'ATE poblacional ≈ 0, pero el efecto vive en la heterogeneidad individual:',
         size=17, color=INK)
    _txt(s, 0.6, 6.25, 12.1, 0.55,
         '4 pressure-clutch leaders y 22 jugadores con respuesta ofensiva significativa post-marcar.',
         size=17, color=INK)
    _footer(s, 2, total)

    # ============ 3. MOTIVACION ============
    s = _blank(prs)
    _section_chip(s, 'Motivación')
    _title(s, 'Por qué este TFM')
    _txt(s, 0.6, 1.85, 12.1, 0.55,
         'El clutch existe en la intuición del aficionado y del scout.',
         size=20, color=INK)
    _txt(s, 0.6, 2.40, 12.1, 0.55,
         'Mbappé marca tres goles en la Final. Lautaro falla un mano-a-mano. Saka se ofrece al penalti.',
         size=17, color=GRAY)
    _txt(s, 0.6, 3.10, 12.1, 0.55,
         'Pero los rankings públicos miden lo que pasa, no la reacción al shock.',
         size=20, color=INK)
    _txt(s, 0.6, 3.65, 12.1, 0.55,
         'Parten de eventing puro, sin tracking sincronizado y sin separar lo individual del bloque.',
         size=17, color=GRAY)
    _txt(s, 0.6, 4.50, 12.1, 0.55,
         'Resultado: un jugador con buena estadística post-gol suele estar',
         size=20, color=INK)
    _txt(s, 0.6, 5.05, 12.1, 0.55,
         'en un equipo que ya estaba dominando el partido.',
         size=20, color=INK)
    _txt(s, 0.6, 6.05, 12.1, 0.65,
         'El clutch verdadero requiere descontar lo que hace su bloque.',
         size=22, color=ORANGE, bold=True)
    _footer(s, 3, total)

    # ============ 4. PREGUNTA + OBJETIVO ============
    s = _blank(prs)
    _section_chip(s, 'Objetivo')
    _title(s, 'Pregunta de investigación y objetivo')
    _txt(s, 0.6, 1.85, 12.1, 0.55, 'Pregunta',
         size=16, color=ORANGE, bold=True)
    _txt(s, 0.6, 2.40, 12.1, 0.55,
         '¿Es la respuesta del jugador al shock emocional del gol',
         size=22, color=NAVY, bold=True)
    _txt(s, 0.6, 2.95, 12.1, 0.55,
         'una propiedad individual estable, residualizada contra el bloque?',
         size=22, color=NAVY, bold=True)
    _txt(s, 0.6, 4.10, 12.1, 0.55, 'Objetivo general',
         size=16, color=ORANGE, bold=True)
    _txt(s, 0.6, 4.65, 12.1, 0.55,
         'Estimar causalmente el efecto del shock sobre el rendimiento individual,',
         size=20, color=INK)
    _txt(s, 0.6, 5.18, 12.1, 0.55,
         'sobre cuatro canales (Ofensivo · Defensivo · Off-ball · Físico)',
         size=20, color=INK)
    _txt(s, 0.6, 5.71, 12.1, 0.55,
         'y tres contextos (post-gol a favor · post-gol en contra · proximidad de eliminación),',
         size=20, color=INK)
    _txt(s, 0.6, 6.24, 12.1, 0.55,
         'con cuantificación de la incertidumbre por jugador.',
         size=20, color=INK)
    _footer(s, 4, total)

    # ============ 5. 6 OBJETIVOS ESPECIFICOS ============
    s = _blank(prs)
    _section_chip(s, 'Objetivo')
    _title(s, 'Seis objetivos específicos')
    oes = [
        ('OE1', 'Backbone temporal: Win Probability bayesiana + leverage + proximidad de eliminación'),
        ('OE2', 'Post-Shot xG calibrado + corpus de near-miss como control cuasi-experimental'),
        ('OE3', 'Cuantificar el rendimiento en cuatro canales independientes'),
        ('OE4', 'Identificación causal dual: DiD within-player + AIPW con cross-fitting'),
        ('OE5', 'CATE jerárquico bayesiano multivariate (NUTS HMC + LKJ cross-canal)'),
        ('OE6', 'Ensamblar tabla scout-facing por jugador, canal y contexto'),
    ]
    for i, (k, v) in enumerate(oes):
        y = 2.05 + i * 0.75
        _txt(s, 0.7, y, 1.0, 0.6, k, size=22, color=NAVY, bold=True)
        _txt(s, 1.9, y+0.05, 10.7, 0.55, v, size=17, color=INK)
    _footer(s, 5, total)

    # ============ 6. TRES FUENTES ============
    s = _blank(prs)
    _section_chip(s, 'Datos')
    _title(s, 'Tres fuentes de datos')
    cols = [
        ('PFF FC',     'World Cup Qatar 2022',  'TARGET',  '64 partidos · eventing + tracking 25 Hz sincronizado · grades humanos por evento · rosters oficiales'),
        ('Wyscout',    '2017/18 open dataset',  'WP BACKBONE', '1941 partidos · cinco grandes ligas europeas + WC 2018 + Euro 2016'),
        ('StatsBomb',  'Open data filtrado',    'PSxG + VAEP', '136 partidos · Euro 2020 + Euro 2024 + Bundesliga 23/24 · incluye 360 freeze-frames'),
    ]
    for i, (k, sub, role, vol) in enumerate(cols):
        y = 2.05 + i * 1.55
        _txt(s, 0.7, y, 3.0, 0.55, k, size=26, color=NAVY, bold=True)
        _txt(s, 3.8, y, 4.0, 0.55, sub, size=15, color=GRAY, italic=True)
        _txt(s, 9.5, y, 3.2, 0.55, role, size=13, color=ORANGE, bold=True, align=PP_ALIGN.RIGHT)
        _txt(s, 0.7, y+0.65, 12.0, 0.75, vol, size=14, color=INK)
        if i < 2:
            _line(s, x=0.7, y=y+1.40, w=11.9, color=SEP, weight_pt=1)
    _footer(s, 6, total)

    # ============ 7. POR QUE PFF ============
    s = _blank(prs)
    _section_chip(s, 'Datos')
    _title(s, 'Por qué PFF FC')
    _txt(s, 0.6, 1.85, 12.1, 0.55,
         'Único corpus público que combina los tres ingredientes estructurales',
         size=20, color=INK)
    _txt(s, 0.6, 2.40, 12.1, 0.55,
         'que la pregunta de investigación exige simultáneamente:',
         size=20, color=INK)
    ings = [
        ('1', 'Tracking continuo full-pitch', 'de los 22 jugadores a 25 Hz'),
        ('2', 'Sincronización exacta',         'tracking ↔ eventing vía game_event.start_frame'),
        ('3', 'Cobertura completa',            'de fase eliminatoria con prórroga incluida'),
    ]
    for i, (n, h, sub) in enumerate(ings):
        y = 3.35 + i * 0.85
        _txt(s, 0.9, y, 0.8, 0.7, n, size=42, color=ORANGE, bold=True)
        _txt(s, 2.0, y+0.05, 4.5, 0.5, h, size=20, color=NAVY, bold=True)
        _txt(s, 6.6, y+0.10, 6.5, 0.5, sub, size=16, color=INK)
    _txt(s, 0.6, 6.10, 12.1, 0.55,
         'Los grades humanos por evento son ingrediente opcional (prior CATE + validación externa), no estructural.',
         size=13, color=GRAY, italic=True)
    _footer(s, 7, total)

    # ============ 8. VOLUMEN ============
    s = _blank(prs)
    _section_chip(s, 'Datos')
    _title(s, 'Corpus de trabajo en cifras')
    cells = [
        ('64',  'partidos',                       'del Mundial 2022'),
        ('172', 'shocks',                         '120 grupos + 52 eliminatoria'),
        ('70',  'near-misses',                    '5 tipologías pre-registradas'),
        ('511', 'jugadores',                      'con minutos en campo'),
        ('233', 'jugadores',                      'con ≥ 270 min (muestra fiable)'),
        ('166', 'jugadores',                      'con grades PFF para validación'),
    ]
    for i, (n, label, sub) in enumerate(cells):
        row, col = divmod(i, 3)
        x = 0.5 + col * 4.30
        y = 2.05 + row * 2.55
        _txt(s, x, y,        4.10, 1.10, n, size=70, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
        _txt(s, x, y+1.20,   4.10, 0.45, label, size=18, color=INK,  bold=True, align=PP_ALIGN.CENTER)
        _txt(s, x, y+1.68,   4.10, 0.45, sub,   size=13, color=GRAY, align=PP_ALIGN.CENTER)
    _footer(s, 8, total)

    # ============ 9. PIPELINE DAG ============
    s = _blank(prs)
    _section_chip(s, 'Pipeline')
    _title(s, 'Pipeline en seis fases')
    _txt(s, 0.6, 1.45, 12.1, 0.45,
         'Extracción → WP backbone → shocks/near-miss → cuatro canales en paralelo → CATE jerárquico → ensamblaje scout-facing',
         size=12, color=GRAY, italic=True)
    _img(s, VIZ / 'fig_cap4_pipeline_dag.jpg', x=0.6, y=1.95, w=12.1)
    _footer(s, 9, total)

    # ============ 10. MAPA CONCEPTUAL ============
    s = _blank(prs)
    _section_chip(s, 'Marco')
    _title(s, 'Mapa conceptual del CCV')
    _txt(s, 0.6, 1.45, 12.1, 0.45,
         'Tipo de shock × ¿el jugador rompe del bloque? × ¿reacción positiva? → proyectado sobre cuatro canales',
         size=12, color=GRAY, italic=True)
    _img(s, VIZ / 'fig_cap4_mapa_conceptual.png', x=0.6, y=2.00, w=12.1)
    _footer(s, 10, total)

    # ============ 11. 5 CAPAS CAUSALES ============
    s = _blank(prs)
    _section_chip(s, 'Marco')
    _title(s, 'Cinco capas de identificación causal')
    _txt(s, 0.6, 1.45, 12.1, 0.45,
         'Cada capa retira un confusor distinto. El estimador causal emerge sólo cuando las cinco operan en secuencia.',
         size=12, color=GRAY, italic=True)
    _img(s, VIZ / 'fig_cap4_capas_causales.png', x=2.65, y=1.95, w=8.0)
    _footer(s, 11, total)

    # ============ 12. DiD ============
    s = _blank(prs)
    _section_chip(s, 'Identificación causal')
    _title(s, 'Diferencias-en-diferencias within-player')
    _txt(s, 0.6, 1.85, 6.3, 0.45, 'Modelo TWFE',
         size=18, color=NAVY, bold=True)
    _txt(s, 0.6, 2.35, 6.3, 0.60,
         'Y = αᵢ,ₛ + γₜ−ₜₛ + τ · 1[t > tₛ] + ε',
         size=18, color=INK, italic=True)
    _txt(s, 0.6, 3.20, 6.3, 0.45, 'Verificaciones independientes',
         size=18, color=NAVY, bold=True)
    bullets = [
        'Sun & Abraham 2021 — sin contaminación entre cohortes',
        'Borusyak-Jaravel-Spiess 2024 — imputación contrafactual',
        'HonestDiD Rambachan-Roth 2023 — cotas de relajación',
        'Test F de pre-tendencias agregadas (Roth 2022)',
    ]
    for i, b in enumerate(bullets):
        _txt(s, 0.85, 3.75 + i*0.45, 6.3, 0.40, '— ' + b, size=14, color=INK)
    _img(s, VIZ / 'event_study.png', x=7.30, y=1.85, w=5.5)
    _footer(s, 12, total)

    # ============ 13. AIPW ============
    s = _blank(prs)
    _section_chip(s, 'Identificación causal')
    _title(s, 'AIPW con cross-fitting (DML)')
    _txt(s, 0.6, 1.85, 12.1, 0.55,
         'Doblemente robusto asintótico bajo tres condiciones:',
         size=20, color=INK)
    cond = [
        'correcta especificación de al menos uno de los nuisance',
        'overlap positivo (0 < ê(X) < 1 con margen)',
        'consistencia n⁻¹ᐟ⁴ de los nuisance (Chernozhukov 2018)',
    ]
    for i, c in enumerate(cond):
        _txt(s, 0.9, 2.65 + i*0.55, 12.0, 0.50, '— ' + c, size=16, color=INK)
    _txt(s, 0.6, 4.55, 12.1, 0.55, 'Implementación',
         size=18, color=NAVY, bold=True)
    impl = [
        'Modelos nuisance: LightGBM, 5-fold estratificado por partido',
        'Propensity score ê(X) + outcome regression μ̂(X) sobre confounders observables',
        'Confounders: minuto, marcador, fuerza del rival, fase del torneo',
    ]
    for i, ii in enumerate(impl):
        _txt(s, 0.9, 5.10 + i*0.50, 12.0, 0.45, '— ' + ii, size=15, color=INK)
    _footer(s, 13, total)

    # ============ 14. CATE BAYESIANO ============
    s = _blank(prs)
    _section_chip(s, 'CATE')
    _title(s, 'CATE jerárquico bayesiano multivariate')
    _txt(s, 0.6, 1.85, 6.3, 0.45, 'Estructura',
         size=18, color=NAVY, bold=True)
    feats = [
        '4 canales conjuntamente (no separados)',
        'Correlaciones cross-canal vía LKJ Cholesky',
        'Priors informados por grades PFF (opcionales)',
        'Parametrización non-centered (Betancourt 2015)',
    ]
    for i, f in enumerate(feats):
        _txt(s, 0.85, 2.35 + i*0.45, 6.3, 0.40, '— ' + f, size=14, color=INK)
    _txt(s, 0.6, 4.55, 6.3, 0.45, 'Sampling',
         size=18, color=NAVY, bold=True)
    samp = [
        'NUTS HMC · 4 chains × 1000 warmup + 1000 sampling',
        'target_accept_prob = 0.95 (geometría LKJ)',
        '0 divergencias · 108 / 144 hyperparams con R-hat < 1.05',
        'PPC 8/8 canales calibrados (KS p-valor > 0.05)',
    ]
    for i, x in enumerate(samp):
        _txt(s, 0.85, 5.05 + i*0.45, 6.3, 0.40, '— ' + x, size=14, color=INK)
    _img(s, VIZ / 'fig_cap6_cate_heterogeneity.png', x=7.30, y=1.85, w=5.5)
    _footer(s, 14, total)

    # ============ 15-18. 4 CANALES ============
    canales = [
        (15, 'Canal Ofensivo',  ORANGE,
            [('atomic-VAEP (CatBoost) sobre representación SPADL', 'AUC scores 0.83 / concedes 0.87'),
             ('un-xPass (Robberechts 2023) — creatividad en pase', 'AUC 0.83'),
             ('Entrenamiento: 456k acciones + 137k pases',         'StatsBomb open data')]),
        (16, 'Canal Defensivo', BLUE_LT,
            [('VDEP estricto (Toda 2022): recovery + attacked',    'AUC 0.80 / 0.83'),
             ('exPress (Lee 2025) — P(recuperación < 5s | press)', 'AUC 0.62'),
             ('Atribución frame-nearest al defensor más cercano',  'Frame level 25 Hz')]),
        (17, 'Canal Off-ball',  NAVY,
            [('OBSO + C-OBSO (Spearman 2018, Teranishi 2022)',     'Off-ball Score Opportunity'),
             ('Pitch Control vectorizado sobre tracking 25 Hz',     'C-OBSO vs grade PFF r = +0.29'),
             ('64 partidos full-time 22 jugadores · 25 Hz',         'n=673, p < 10⁻¹⁴')]),
        (18, 'Canal Físico',    GRAY,
            [('Protocolo Bradley 2024 sobre velocidades suavizadas','Filtro Butterworth fase cero'),
             ('Cap a 11 m/s preservando dirección',                 '64 partidos a 25 Hz'),
             ('Residualización bayesiana jerárquica via SVI',        'Línea base personal por curva minuto')]),
    ]
    for n, title, color_acc, rows in canales:
        s = _blank(prs)
        _section_chip(s, 'Canales')
        _title(s, title)
        # encabezado tabla
        _txt(s, 0.7, 1.95, 8.5, 0.45, 'Building block', size=13, color=GRAY, bold=True)
        _txt(s, 9.3, 1.95, 3.6, 0.45, 'Métrica / Validación', size=13, color=GRAY, bold=True)
        _line(s, x=0.7, y=2.35, w=12.0, color=SEP, weight_pt=1)
        for i, (bb, met) in enumerate(rows):
            y = 2.50 + i * 1.10
            _txt(s, 0.7, y, 8.5, 0.55, bb, size=17, color=INK)
            _txt(s, 9.3, y, 3.6, 0.55, met, size=15, color=color_acc, bold=True)
            if i < len(rows) - 1:
                _line(s, x=0.7, y=y+0.95, w=12.0, color=SEP, weight_pt=1)
        _footer(s, n, total)

    # ============ 19. PSxG ANCLA ============
    s = _blank(prs)
    _section_chip(s, 'Validación')
    _title(s, 'PSxG: ancla técnica del corpus near-miss')
    _txt(s, 0.6, 1.85, 6.3, 0.45,
         'Modelo predictivo',
         size=18, color=NAVY, bold=True)
    _txt(s, 0.85, 2.35, 6.3, 0.45,
         'LightGBM + Optuna (60 iter) + isotonic',
         size=15, color=INK)
    _txt(s, 0.85, 2.83, 6.3, 0.45,
         'Features: end_y, end_z, configuración 360 freeze-frame',
         size=15, color=INK)
    _txt(s, 0.85, 3.31, 6.3, 0.45,
         'Train: StatsBomb open data fuera del Mundial',
         size=15, color=INK)
    _txt(s, 0.6, 4.10, 6.3, 0.45, 'Métricas', size=18, color=NAVY, bold=True)
    metrics = [
        ('AUC OOF',          '0.968'),
        ('AUC holdout WC22', '0.976'),
        ('vs StatsBomb xG (pre-disparo)',  '0.844'),
        ('ECE holdout',      '0.011'),
        ('Brier holdout',    '0.037'),
    ]
    for i, (k, v) in enumerate(metrics):
        y = 4.60 + i * 0.40
        _txt(s, 0.85, y, 4.5, 0.40, k, size=13, color=INK)
        _txt(s, 5.35, y, 1.5, 0.40, v, size=14, color=ORANGE, bold=True, align=PP_ALIGN.RIGHT)
    _img(s, VIZ / 'fig_cap6_psxg_calibration.png', x=7.30, y=1.85, w=5.5)
    _footer(s, 19, total)

    # ============ 20. FICHA MBAPPE ============
    s = _blank(prs)
    _section_chip(s, 'Output scout-facing')
    _title(s, 'Ficha CCV — Kylian Mbappé')
    _txt(s, 0.6, 1.45, 12.1, 0.45,
         'Pressure-clutch leader del torneo · CATE +0.110 desviaciones estándar · P(β > 0) = 0.97',
         size=14, color=ORANGE, bold=True, align=PP_ALIGN.LEFT)
    _img(s, VIZ / 'radar_report_3870.png', x=1.0, y=2.00, w=11.3)
    _footer(s, 20, total)

    # ============ 21. SCATTER REMONTADOR x CERROJO ============
    s = _blank(prs)
    _section_chip(s, 'Output scout-facing')
    _title(s, 'Remontador × Cerrojo — los 511 jugadores')
    _txt(s, 0.6, 1.45, 12.1, 0.45,
         'Dos índices ortogonales: respuesta al gol en contra (Remontador) vs. respuesta al gol a favor (Cerrojo).',
         size=12, color=GRAY, italic=True)
    _img(s, VIZ / 'scatter_remontador_cerrojo.png', x=1.7, y=1.95, w=9.9)
    _footer(s, 21, total)

    # ============ 22. ATE = 0 ============
    s = _blank(prs)
    _section_chip(s, 'Resultados')
    _title(s, 'ATE poblacional indistinguible de cero')
    _img(s, VIZ / 'fig_cap6_window_sensitivity.png', x=0.6, y=1.85, w=8.0)
    # texto al lado derecho
    _txt(s, 8.9, 1.95, 4.0, 0.50, 'Lectura',
         size=18, color=NAVY, bold=True)
    _txt(s, 8.9, 2.45, 4.2, 0.55,
         '8 celdas canal × contexto.',
         size=15, color=INK)
    _txt(s, 8.9, 2.92, 4.2, 0.55,
         'Ninguna cruza la significatividad al 5%.',
         size=15, color=INK)
    _txt(s, 8.9, 3.85, 4.0, 0.50, 'Robustez', size=18, color=NAVY, bold=True)
    _txt(s, 8.9, 4.35, 4.2, 0.45,
         'Placebo 1000 perm + BH FDR:',
         size=13, color=GRAY, bold=True)
    _txt(s, 8.9, 4.75, 4.2, 0.45,
         'ningún canal sale del null (p_FDR=0.98)',
         size=13, color=INK)
    _txt(s, 8.9, 5.25, 4.2, 0.45,
         'HonestDiD:',
         size=13, color=GRAY, bold=True)
    _txt(s, 8.9, 5.65, 4.2, 0.45,
         'cotas contienen el cero en los 3 niveles',
         size=13, color=INK)
    _txt(s, 8.9, 6.30, 4.2, 0.55,
         'El efecto medio no se',
         size=15, color=ORANGE, bold=True)
    _txt(s, 8.9, 6.60, 4.2, 0.55,
         'distingue del ruido.',
         size=15, color=ORANGE, bold=True)
    _footer(s, 22, total)

    # ============ 23. 4 LEADERS ============
    s = _blank(prs)
    _section_chip(s, 'Resultados')
    _title(s, 'Cuatro pressure-clutch leaders identificados')
    _txt(s, 0.6, 1.45, 12.1, 0.45,
         'CATE pressure-clutch · probabilidad posterior |P(β)| ≥ 0.85 · sobre 511 jugadores evaluados',
         size=12, color=GRAY, italic=True)
    headers = ['Jugador', 'Selección', 'CATE', 'P(β > 0)']
    rows = [
        ('Kylian Mbappé',    'Francia',    '+0.110', '0.97', BLUE_LT),
        ('Marcus Rashford',  'Inglaterra', '+0.050', '0.89', BLUE_LT),
        ('Bukayo Saka',      'Inglaterra', '+0.050', '0.86', BLUE_LT),
        ('Mohammed Muntari', 'Catar',      '−0.055', '0.12', ORANGE),
    ]
    x_cols = [0.7, 4.7, 8.3, 10.7]
    w_cols = [4.0, 3.6, 2.4, 2.0]
    # header
    y_h = 2.30
    for i, h in enumerate(headers):
        _txt(s, x_cols[i], y_h, w_cols[i], 0.45, h,
             size=14, color=GRAY, bold=True)
    _line(s, x=0.7, y=2.75, w=12.0, color=SEP, weight_pt=1)
    # rows
    for j, (name, team, cate, prob, col) in enumerate(rows):
        y = 3.00 + j * 0.85
        _txt(s, x_cols[0], y, w_cols[0], 0.60, name, size=20, color=INK, bold=True)
        _txt(s, x_cols[1], y, w_cols[1], 0.60, team, size=18, color=INK)
        _txt(s, x_cols[2], y, w_cols[2], 0.60, cate, size=22, color=col, bold=True)
        _txt(s, x_cols[3], y, w_cols[3], 0.60, prob, size=22, color=col, bold=True)
        if j < 3:
            _line(s, x=0.7, y=y+0.75, w=12.0, color=SEP, weight_pt=1)
    _txt(s, 0.6, 6.65, 12.1, 0.40,
         'Muntari es el único negativo significativo: cae bajo presión de eliminación inminente.',
         size=12, color=GRAY, italic=True, align=PP_ALIGN.CENTER)
    _footer(s, 23, total)

    # ============ 24. 22 OFENSIVO ============
    s = _blank(prs)
    _section_chip(s, 'Resultados')
    _title(s, 'Ofensivo-tras-marcar: 22 jugadores significativos')
    _txt(s, 0.6, 1.45, 12.1, 0.45,
         'La celda canal × contexto con más señal individual del torneo.',
         size=14, color=GRAY, italic=True)
    # dos columnas: 5 al alza / 17 a la baja
    _txt(s, 0.6, 2.40, 6.2, 1.4, '5', size=110, color=BLUE_LT, bold=True, align=PP_ALIGN.CENTER)
    _txt(s, 0.6, 4.10, 6.2, 0.55, 'al alza', size=24, color=INK, bold=True, align=PP_ALIGN.CENTER)
    _txt(s, 0.6, 4.70, 6.2, 0.45, 'ascienden producción ofensiva', size=14, color=GRAY, align=PP_ALIGN.CENTER)
    _txt(s, 0.6, 5.15, 6.2, 0.45, 'tras marcar gol', size=14, color=GRAY, align=PP_ALIGN.CENTER)

    _txt(s, 6.5, 2.40, 6.2, 1.4, '17', size=110, color=ORANGE, bold=True, align=PP_ALIGN.CENTER)
    _txt(s, 6.5, 4.10, 6.2, 0.55, 'a la baja', size=24, color=INK, bold=True, align=PP_ALIGN.CENTER)
    _txt(s, 6.5, 4.70, 6.2, 0.45, 'se relajan tras marcar', size=14, color=GRAY, align=PP_ALIGN.CENTER)
    _txt(s, 6.5, 5.15, 6.2, 0.45, '(efecto "ya está hecho")', size=14, color=GRAY, align=PP_ALIGN.CENTER)

    _line(s, x=2.0, y=6.05, w=9.3, color=SEP, weight_pt=1)
    _txt(s, 0.6, 6.20, 12.1, 0.55,
         'Signos opuestos en magnitudes similares → el promedio cancela y el ATE colapsa a cero.',
         size=17, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    _footer(s, 24, total)

    # ============ 25. CONCLUSION CLAVE ============
    s = _blank(prs)
    _section_chip(s, 'Conclusión')
    _line(s, x=4.5, y=1.45, w=4.3, color=ORANGE, weight_pt=4)
    _txt(s, 0.6, 1.95, 12.1, 1.10,
         'ATE nulo no implica efecto nulo.',
         size=46, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    _txt(s, 0.6, 3.30, 12.1, 0.65,
         'El clutch vive en la heterogeneidad individual,',
         size=28, color=BLUE_LT, align=PP_ALIGN.CENTER)
    _txt(s, 0.6, 3.95, 12.1, 0.65,
         'no en la media poblacional.',
         size=28, color=BLUE_LT, align=PP_ALIGN.CENTER)
    _line(s, x=5.6, y=4.95, w=2.1, color=SEP, weight_pt=1)
    _txt(s, 0.6, 5.30, 12.1, 0.55,
         'El framework permite identificar QUIÉN responde al shock,',
         size=18, color=INK, align=PP_ALIGN.CENTER)
    _txt(s, 0.6, 5.85, 12.1, 0.55,
         'en qué canal, en qué contexto, y con cuánta certeza.',
         size=18, color=INK, align=PP_ALIGN.CENTER)
    _footer(s, 25, total)

    # ============ 26. LIMITACIONES + FUTURO ============
    s = _blank(prs)
    _section_chip(s, 'Cierre')
    _title(s, 'Limitaciones y líneas futuras')
    # Limitaciones (izquierda)
    _txt(s, 0.7, 1.95, 6.0, 0.50, 'Limitaciones', size=20, color=ORANGE, bold=True)
    lims = [
        'Un único torneo (Mundial Qatar 2022)',
        '172 shocks → CATE individual fiable pero no por partido',
        '70 near-misses → triangulación agregada, no por jugador',
        'Building blocks distintos a VAEP son reimplementación propia',
    ]
    for i, b in enumerate(lims):
        _txt(s, 0.9, 2.65 + i*0.85, 5.9, 0.80, '— ' + b, size=15, color=INK)
    # Líneas futuras (derecha)
    _txt(s, 7.0, 1.95, 6.0, 0.50, 'Líneas futuras', size=20, color=BLUE_LT, bold=True)
    fut = [
        'Más torneos KO (Eurocopa, Champions, World Cup 2026)',
        'Corpus near-miss multitorneo para verificación por jugador',
        'Catálogo de shock ampliado (sustituciones, sup./inf. numérica)',
        'Portabilidad a "clutch del bloque" a nivel equipo',
    ]
    for i, b in enumerate(fut):
        _txt(s, 7.2, 2.65 + i*0.85, 5.9, 0.80, '— ' + b, size=15, color=INK)
    _footer(s, 26, total)

    # ============ 27. GRACIAS ============
    s = _blank(prs)
    _line(s, x=4.5, y=1.55, w=4.3, color=ORANGE, weight_pt=4)
    _txt(s, 0.6, 1.95, 12.1, 1.20,
         'Gracias',
         size=82, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    _txt(s, 0.6, 3.65, 12.1, 0.55,
         'Documento TFM completo + pipeline E2E reproducible:',
         size=18, color=INK, align=PP_ALIGN.CENTER)
    _txt(s, 0.6, 4.25, 12.1, 0.65,
         'github.com/jaime-oriol/CCV',
         size=26, color=BLUE_LT, bold=True, align=PP_ALIGN.CENTER)
    _img(s, LOGO, x=5.42, y=5.30, w=2.50)
    _txt(s, 0.6, 7.05, 12.1, 0.35,
         'Jaime Oriol Goicoechea · Máster Big Data Aplicado Scouting Deportivo · Sports Data Campus · 2026',
         size=10, color=GRAY, align=PP_ALIGN.CENTER)

    prs.save(str(OUT))
    print(f'guardado: {OUT.relative_to(REPO)}')
    print(f'  slides: {len(prs.slides)} · size {prs.slide_width/914400:.2f} x {prs.slide_height/914400:.2f}')


if __name__ == '__main__':
    build()
