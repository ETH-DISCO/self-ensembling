.PHONY: install
install:
	sudo tlmgr update --self
	sudo tlmgr install enumitem fontawesome5 adjustbox tcolorbox tikzfill pdfcol listingsutf8

.PHONY: pdf
pdf:
	pdflatex -interaction=nonstopmode resume.tex
	rm -f *.aux *.log *.out *.synctex.gz
	# open -a "Google Chrome" resume.pdf

.PHONY: up
up:
	git pull
	git add .
	git commit -m "up"
	git push
