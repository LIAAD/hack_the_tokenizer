cd "/home/yali/MEGA/Thesis Document"
cd Front
xelatex -interaction=nonstopmode -file-line-error main
xelatex -interaction=nonstopmode -file-line-error main

# Clean up auxiliary files
rm -f *.toc *.aux *.out *.bbl *.blg *.lof *.lot *.nav *.snm *.vrb *.synctex.gz *.fdb_latexmk *.fls *.xdv *.acn *.brf *.bglo *.glsdefs *.ist *.glo

cd ..
xelatex -interaction=nonstopmode -file-line-error main  # Generating the auxiliary files
xelatex -interaction=nonstopmode -file-line-error main  # This is to use the auxiliary files in the generation


# Clean up auxiliary files
rm -f *.toc  *.aux *.out *.bbl *.blg *.lof *.lot *.nav *.snm *.vrb *.synctex.gz *.fdb_latexmk *.fls *.xdv *.acn *.brf *.bglo *.glsdefs *.ist *.glo

# Optional: Alternative cleanup command
# find . -type f ! -name 'main.tex' ! -name 'main.pdf' ! -name '*.sh' -delete