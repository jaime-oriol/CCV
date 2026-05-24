# Makefile TFM doc — compilacion LaTeX
# Uso:
#   make            -> compila main.pdf en build/
#   make watch      -> recompila al guardar (latexmk -pvc)
#   make clean      -> borra build/ entero
#   make open       -> abre build/main.pdf

.PHONY: pdf watch clean open

pdf:
	latexmk -pdf -interaction=nonstopmode main.tex

watch:
	latexmk -pdf -pvc -interaction=nonstopmode main.tex

clean:
	latexmk -C
	rm -rf build/

open:
	@xdg-open build/main.pdf 2>/dev/null || open build/main.pdf 2>/dev/null || echo "abre build/main.pdf manualmente"
