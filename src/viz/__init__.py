"""src.viz - Capa de visualizacion del PCJ.

Estilo dark (#313332), paleta ATT/DEF, colormap PPCF y percentil propios,
logo JO Jaime Oriol como firma.

  common         estilo, colores, colormaps, draw_pitch, helpers
  ppcf           superficie PPCF sobre el campo (Z02 + tracking PFF)
  scatter        2 diamond globales: Remontador x Cerrojo + ataque marcar/presion (511 jug)
  scatter_team   2 diamond por seleccion con caras (FotMob) como markers
  radar          radar geometrico de 8 ejes (CATEs canal x shock)
  radar_report   radar + tabla percentil lado a lado (ficha scout completa)
  figures        event-study causal M12 + utilidades
  __main__       runner: python -m src.viz [radar|report] <query>
"""
