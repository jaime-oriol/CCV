"""src.viz - Capa de visualizacion del PCJ con la identidad visual propia.

Estilo "Diagonality" portado tal cual de los repos jaime-oriol/Diagonality_3D
y jaime-oriol/pitch-control: fondo #313332, paleta ATT/DEF, colormaps
PPCF/percentil, logo Diagonality discreto abajo-derecha.

  common  - estilo, colores, colormaps, draw_pitch, helpers _logo/_style/_save
  ppcf    - superficie PPCF sobre el campo (consume Z02 + tracking PFF)
  scatter - scatter chasing x protecting (consume pcj_table)
  radar   - ficha PCJ: radar + tabla percentil (consume pcj_table)
  figures - figuras analiticas (event-study M12, ...)
"""
