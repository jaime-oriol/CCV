"""src.viz - Capa de visualizacion del xCV.

Identidad LIGHT OPTA-STYLE: fondo blanco, textos negros, tipografia Chakra Petch,
paleta ATT/DEF light, PCT_CMAP morado→fuchsia→rosa, logo JO como firma.

  common         estilo, colores, cmaps, draw_pitch, draw_header, add_logo
  ppcf           superficie Pitch Control sobre el campo (Z02 + tracking PFF)
  scatter        2 scatter globales: Remontador x Cerrojo + ataque tras marcar / bajo presion
  scatter_team   2 scatter por seleccion con caras FotMob (resto del torneo en gris)
  radar          radar geometrico 8 o 12 ejes (CATEs canal x contexto)
  radar_report   radar + tabla percentil lado a lado (ficha scout completa)
  figures        event-study causal (M12, figura de validacion / metodos)
  __main__       runner: python -m src.viz [radar|report] <query>
"""
